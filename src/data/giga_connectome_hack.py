import argparse, glob, os
from pathlib import Path

import h5py
import nibabel as nib
from nilearn.connectome import ConnectivityMeasure
from nilearn.image import (
    resample_to_img,
    new_img_like,
    get_data,
    math_img,
    load_img,
)
from nilearn.maskers import NiftiLabelsMasker, NiftiMapsMasker
from nilearn.masking import compute_multi_epi_mask
from scipy.ndimage import binary_closing
from tqdm import tqdm

from giga_connectome import __version__
from giga_connectome import (
    generate_gm_mask_atlas,
    load_atlas_setting,
    run_postprocessing_dataset,
    get_denoise_strategy,
)
from giga_connectome.connectome import generate_timeseries_connectomes
from giga_connectome.denoise import is_ica_aroma, denoise_nifti_voxel
from giga_connectome.mask import _check_mask_affine, _get_consistent_masks
from giga_connectome import utils

"""
This script is a custom adapatation of the giga_connectome library
which extracts denoised time series and computes connectomes from
fMRI BOLD data using brain parcellations.

Source:
https://github.com/SIMEXP/giga_connectome/tree/main
"""
# grey matter group mask is only supplied in MNI152NLin2009c(A)sym
STUDY_PARAMS = {
    "template": "MNI152NLin2009cAsym",
    "templateflow_dir": "atlases/atlas-MIST",
    "gm_res": "02",
    "n_iter": 2,
    "desc": "444",
    "atlas_name": "MIST",
    "atlas_type": "dseg",
    "calcul_avgcorr": False,
}


def get_arguments(argv=None):
    """Entry point."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "Generate denoised timeseries and Pearson's correlation based "
            "connectomes from fmriprep processed dataset."
        ),
    )
    parser.add_argument(
        "work_dir",
        action="store",
        type=Path,
        help="The directory with the source code and input dataset "
        "(e.g. fMRIPrep derivative) formatted according to the BIDS standard.",
    )
    parser.add_argument(
        "output_dir",
        action="store",
        type=Path,
        help="The directory where the output files should be stored.",
    )
    parser.add_argument(
        "-v", "--version", action="version", version=__version__
    )
    parser.add_argument(
        "--participant_label",
        help="The label(s) of the participant(s) that should be analyzed. The "
        "label corresponds to sub-<participant_label> from the BIDS spec (so "
        "it does not include 'sub-'). If this parameter is not provided all "
        "subjects should be analyzed. Multiple participants can be specified "
        "with a space separated list.",
        nargs="+",
    )
    parser.add_argument(
        "--denoise-strategy",
        help="The choice of post-processing for denoising. The default "
        "choices are: 'simple', 'simple+gsr', 'scrubbing.2', "
        "'scrubbing.2+gsr', 'scrubbing.5', 'scrubbing.5+gsr', 'acompcor50', "
        "'icaaroma'. User can pass a path to a json file containing "
        "configuration for their own choice of denoising strategy. The default"
        "is 'simple'.",
        default="simple",
    )
    parser.add_argument(
        "--standardize",
        help="The choice of signal standardization. The choices are z score "
        "or percent signal change (psc). The default is 'zscore'.",
        choices=["zscore", "psc"],
        default="zscore",
    )
    parser.add_argument(
        "--smoothing_fwhm",
        help="Size of the full-width at half maximum in millimeters of "
        "the spatial smoothing to apply to the signal. The default is 5.0.",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--compression",
        type=str,
        default=None,
        choices=[None, "gzip", "lzf"],
        help="Lossless compression applied to time series in .h5 file. Default is none.",
    )
    parser.add_argument(
        "--compression_opts",
        type=int,
        default=4,
        choices=range(0, 10),
        help="Frame compression level in .h5 file. Value = [0-9]. "
        "Only for lossless gzip compression.",
    )

    return parser.parse_args(argv)


def workflow_hack(args):

    print(vars(args))
    # set file paths
    bids_dir = f"{args.work_dir}/data/friends.fmriprep2"
    output_dir = args.output_dir
    # check output path
    output_dir.mkdir(parents=True, exist_ok=True)

    standardize = utils.parse_standardize_options(args.standardize)
    smoothing_fwhm = args.smoothing_fwhm
    subjects = utils.get_subject_lists(args.participant_label, bids_dir)
    strategy = get_denoise_strategy(args.denoise_strategy)

    # currently we only support the fmriprep defaults
    template = STUDY_PARAMS["template"]
    mask_dir = Path(f"{output_dir}/subject_masks/tpl-{template}")
    mask_dir.mkdir(exist_ok=True, parents=True)

    # preprocessed data do not need high resolution
    gm_res = STUDY_PARAMS["gm_res"]
    n_iter = STUDY_PARAMS["n_iter"]
    templateflow_dir = Path(f'{args.work_dir}/{STUDY_PARAMS["templateflow_dir"]}')
    mni_gm_path = (
        f"{templateflow_dir}/tpl-{template}/"
        f"tpl-{template}_res-{gm_res}_label-GM_probseg.nii.gz",
        )

    desc = STUDY_PARAMS["desc"]
    atlas_name = STUDY_PARAMS["atlas_name"]
    atlas_type = STUDY_PARAMS["atlas_type"]

    parcellation_name = (
        f"{templateflow_dir}/tpl-MNI152NLin2009bAsym/"
        f"tpl-MNI152NLin2009bAsym_res-03_atlas-BASC_desc-{desc}_{atlas_type}.nii.gz"
    )

    calculate_average_correlation = STUDY_PARAMS["calcul_avgcorr"]

    compression_args = {}
    if args.compression is not None:
        compression_args["compression"] = args.compression
        if args.compression == "gzip":
            compression_args["compression_opts"] = args.compression_opts

    for subject in subjects:
        '''
        Compile list of subject's bold and brain mask files
        '''
        found_mask_list = sorted(
            glob.glob(
                f"{bids_dir}/sub-{subject}/"
                f"ses-0*/func/*{template}_"
                "desc-brain_mask.nii.gz",
                ),
            )
        if exclude := _check_mask_affine(found_mask_list, verbose=2):
            found_mask_list, __annotations__ = _get_consistent_masks(
                found_mask_list,
                exclude,
                )
            print(f"Remaining: {len(found_mask_list)} masks")

        bold_list = []
        mask_list = []
        for fm in found_mask_list:
            sub, ses, task, space, ftype, appendix = fm.split('/')[-1].split('_')
            bpath = f"{bids_dir}/{sub}/{ses}/func/{sub}_{ses}_{task}_{space}_desc-preproc_bold.nii.gz"

            if Path(bpath).exists():
                bold_list.append(bpath)
                mask_list.append(fm)

        '''
        Generate subject-specific grey matter mask from all found sessions
        '''
        # grey matter multi-session subject mask in MNI152NLin2009cAsym
        subject_epi_mask = compute_multi_epi_mask(
            mask_list,
            lower_cutoff=0.2,
            upper_cutoff=0.85,
            connected=True,
            opening=False,  # we should be using fMRIPrep masks
            threshold=0.5,
            target_affine=None,
            target_shape=None,
            exclude_zeros=False,
            n_jobs=1,
            memory=None,
            verbose=0,
        )

        print(
            f"Group EPI mask affine:\n{subject_epi_mask.affine}"
            f"\nshape: {subject_epi_mask.shape}"
        )

        mni_gm = nib.squeeze_image(
            resample_to_img(
                source_img=mni_gm_path,
                target_img=subject_epi_mask,
                interpolation="continuous",
            ),
        )

        # the following steps are adapted from
        # nilearn.images.fetch_icbm152_brain_gm_mask
        mni_gm_data = get_data(mni_gm)
        # this is a probalistic mask, getting one fifth of the values
        mni_gm_mask = (mni_gm_data > 0.2).astype("int8")
        mni_gm_mask = binary_closing(mni_gm_mask, iterations=n_iter)
        mni_gm_mask_img = new_img_like(mni_gm, mni_gm_mask)

        # combine both masks into one
        subject_mask_nii = math_img(
            "img1 & img2",
            img1=subject_epi_mask,
            img2=mni_gm_mask_img,
            )

        current_file_name = (
            f"tpl-{template}_res-dataset_label"
            f"-GM_desc-sub-{subject}_mask.nii.gz"
        )
        subject_mask = f"{mask_dir}/{current_file_name}"
        nib.save(subject_mask_nii, subject_mask)

        '''
        Resample parcellation atlas to subject grey matter mask
        '''
        parcellation = nib.load(parcellation_name)
        parcellation_resampled = resample_to_img(
            #parcellation_name, subject_mask, interpolation="nearest"  # load from file paths
            parcellation, subject_mask_nii, interpolation="nearest"
        )
        filename = (
            f"tpl-{template}_"
            f"atlas-{atlas_name}_"
            "res-dataset_"
            f"desc-{desc}_"
            f"{atlas_type}_sub-{subject}.nii.gz"
        )
        save_parcel_path = mask_dir / filename
        nib.save(parcellation_resampled, save_parcel_path)

        '''
        Generate run-level time series
        '''
        atlas_path = Path(save_parcel_path)

        if atlas_type == "dseg":
            atlas_masker = NiftiLabelsMasker(
                labels_img=atlas_path, standardize=False
            )
        elif atlas_type == "probseg":
            atlas_masker = NiftiMapsMasker(maps_img=atlas_path, standardize=False)

        correlation_measure = ConnectivityMeasure(
            kind="correlation", vectorize=False, discard_diagonal=False
        )

        connectome_dir = f"{output_dir}/timeseries"
        Path(connectome_dir).mkdir(parents=True, exist_ok=True)
        connectome_path = (
            f"{connectome_dir}/"
            f"sub-{subject}_friends_BOLDtimeseries_atlas-{atlas_name}{desc}"
            f"_desc-{strategy['name']}.h5"
        )

        processed_episode_list = []

        if Path(connectome_path).exists():
            with h5py.File(connectome_path, 'r') as f:
                processed_episode_list = [f for f in f.keys()]

        for img in tqdm(bold_list):
            try:
                # parse file name
                sub, ses, task, space, ftype, appendix = img.split('/')[-1].split('_')
                print(sub, task)

                if not f"{task.split('-')[-1]}" in processed_episode_list:
                    # process timeseries
                    denoised_img = denoise_nifti_voxel(
                        strategy, subject_mask, standardize, smoothing_fwhm, img
                    )

                    attribute_name = f"{sub}_{ses}_{task}_atlas-{atlas_name}_desc-{desc}"

                    if not denoised_img:
                        time_series_atlas, correlation_matrix = None, None
                        print(f"{attribute_name}: no volume after scrubbing")
                        continue

                    # extract timeseries and connectomes
                    (
                        correlation_matrix,
                        time_series_atlas,
                    ) = generate_timeseries_connectomes(
                        atlas_masker,
                        denoised_img,
                        subject_mask,
                        correlation_measure,
                        calculate_average_correlation,
                    )

                    # Remove last TR from run that stopped one run short for sub-06
                    if f"{task.split('-')[-1]}" == 's01e12a':
                        time_series_atlas = time_series_atlas[:471, :]

                    # dump to h5
                    flag =  "a" if Path(connectome_path).exists() else "w"
                    with h5py.File(connectome_path, flag) as f:
                        if not f"{task.split('-')[-1]}" in f.keys():
                            group = f.create_group(f"{task.split('-')[-1]}")

                            timeseries_dset = group.create_dataset(
                                "timeseries",
                                data=time_series_atlas,
                                **compression_args,
                            )
                            timeseries_dset.attrs["RepetitionTime"] = 1.49

            except:
                print(f"could not process file {img}" )


if __name__ == "__main__":
    """
    Process fMRIPrep outputs to timeseries based on denoising strategy.
    """
    args = get_arguments()
    print(vars(args))

    workflow_hack(args)
