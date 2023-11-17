import argparse
import glob
import os
from pathlib import Path

import h5py
import nibabel as nib
from giga_connectome import (
    __version__,
    get_denoise_strategy,
    utils,
)
from giga_connectome.connectome import generate_timeseries_connectomes
from giga_connectome.denoise import denoise_nifti_voxel, is_ica_aroma
from giga_connectome.mask import _check_mask_affine, _get_consistent_masks
from nilearn.connectome import ConnectivityMeasure
from nilearn.image import (
    get_data,
    math_img,
    new_img_like,
    resample_to_img,
)
from nilearn.maskers import NiftiLabelsMasker, NiftiMapsMasker
from nilearn.masking import compute_multi_epi_mask
from scipy.ndimage import binary_closing
from tqdm import tqdm

"""
This script is a custom adapatation of the giga_connectome library
which extracts denoised time series and computes connectomes from
fMRI BOLD data using brain parcellations.

It requires giga_connectome insalled as a library.

Source:
https://github.com/SIMEXP/giga_connectome/tree/main
"""
# grey matter group mask is only supplied in MNI152NLin2009c(A)sym
STUDY_PARAMETERS = {
    "template": "MNI152NLin2009cAsym",
    "gm_res": "02",
    "n_iter": 2,
    "desc": "444",
    "atlas_name": "MIST",
    "atlas_type": "dseg",
    "calcul_avgcorr": False,
}


def get_arguments(argv=None) -> argparse.Namespace:
    """Entry point."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "Generate denoised timeseries in subject-specific atlas-defined "
            "parcels from fmriprep processed dataset."
        ),
    )
    parser.add_argument(
        "--bids_dir",
        action="store",
        type=Path,
        help="The directory with the input dataset (fMRIPrep derivatives).",
    )
    parser.add_argument(
        "--template_dir",
        action="store",
        type=Path,
        help="The directory with the grey matter template and parcellation "
        "atlas.",
    )
    parser.add_argument(
        "--output_dir",
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
        "subjects will be analyzed. Multiple participants can be specified "
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
        help="Lossless compression applied to time series in .h5 file. "
        "Default is None.",
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


@dataclass
class Study_Params:
    """.

    Dataclass for paths and analysis parameters.
    """

    args: argparse.Namespace
    template: str
    gm_res: int
    n_iter: int
    desc: str
    atlas_name: str
    atlas_type: str
    calcul_avgcorr: bool

    def __post_init__(self):
        # define subject list
        self.subjects = utils.get_subject_lists(
            self.args.participant_label,
            self.args.bids_dir,
        )
        # define analysis params
        self.standardize = utils.parse_standardize_options(
            self.args.standardize,
        )
        self.strategy = get_denoise_strategy(
            self.args.denoise_strategy,
        )
        self.correlation_measure = ConnectivityMeasure(
            kind="correlation", vectorize=False, discard_diagonal=False
        )

        # create output directory
        self.args.output_dir.mkdir(parents=True, exist_ok=True)

        # path to MNI-space grey matter mask
        # currently, giga_connectome only supports the fmriprep defaults
        self.mni_gm_path = (
            f"{self.args.template_dir}/tpl-{self.template}/"
            f"tpl-{self.template}_res-{self.gm_res}_label-GM_probseg.nii.gz",
            )

        # create dir to save subject-specific grey matter mask and parcel atlas
        self.mask_dir = Path(
            f"{self.args.output_dir}/subject_masks/tpl-{self.template}"
        )
        self.mask_dir.mkdir(exist_ok=True, parents=True)

        # create dir to save subject-specific timeseries
        self.timeseries_dir = f"{self.args.output_dir}/timeseries"
        Path(self.timeseries_dir).mkdir(parents=True, exist_ok=True)

        # define path to parcellation atlas
        self.parcellation_path = (
            f"{self.args.template_dir}/tpl-MNI152NLin2009bAsym/"
            "tpl-MNI152NLin2009bAsym_res-03_atlas-BASC_desc"
            f"-{self.desc}_{self.atlas_type}.nii.gz"
        )

        # define compression parameters for timeseries saved in .h5 file
        self.comp_args = {}
        if self.args.compression is not None:
            self.comp_args["compression"] = self.args.compression
            if self.args.compression == "gzip":
                self.comp_args["compression_opts"] = self.args.compression_opts


def compile_bold_list(
    subject: str,
    sp: Study_Params,
) -> (list, list):
    """.

    Compile list of subject's bold and brain mask files.
    """
    found_mask_list = sorted(
        glob.glob(
            f"{sp.args.bids_dir}/sub-{subject}/"
            f"ses-0*/func/*{sp.template}_"
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
        sub, ses, task, space, _, _ = fm.split('/')[-1].split('_')
        bpath = (
            f"{sp.args.bids_dir}/{sub}/{ses}/func/"
            f"{sub}_{ses}_{task}_{space}_desc-preproc_bold.nii.gz"
        )

        if Path(bpath).exists():
            bold_list.append(bpath)
            mask_list.append(fm)

    return bold_list, mask_list


def merge_masks(
    subject_epi_mask: nib.nifti1.Nifti1Image,
    mni_gm_path: str,
    n_iter: int,
)-> nib.nifti1.Nifti1Image:
    """.

    Combine both subject's epi mask and template mask into one GM mask.
    """
    # resample MNI grey matter template mask to subject's grey matter mask
    mni_gm = nib.squeeze_image(
        resample_to_img(
            source_img=mni_gm_path,
            target_img=subject_epi_mask,
            interpolation="continuous",
        ),
    )

    # steps adapted from nilearn.images.fetch_icbm152_brain_gm_mask
    mni_gm_mask = (get_data(mni_gm) > 0.2).astype("int8")
    mni_gm_mask = binary_closing(mni_gm_mask, iterations=n_iter)
    mni_gm_mask_nii = new_img_like(mni_gm, mni_gm_mask)

    # combine both subject and template masks into one
    subject_mask_nii = math_img(
        "img1 & img2",
        img1=subject_epi_mask,
        img2=mni_gm_mask_nii,
    )

    return subject_mask_nii


def make_subjectGM_mask(
    subject: str,
    mask_list: list,
    sp: Study_Params,
) -> (nib.nifti1.Nifti1Image, str):
    """.

    Generate subject-specific grey matter mask from all runs.
    """
    # generate multi-session grey matter subject mask in MNI152NLin2009cAsym
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

    # merge mask from subject's epi files w template grey matter mask
    subject_mask_nii = merge_masks(
        subject_epi_mask,
        sp.mni_gm_path,
        sp.n_iter,
    )

    # save subject grey matter mask
    subject_mask_path = (
        f"{sp.mask_dir}/tpl-{sp.template}_"
        f"res-dataset_label-GM_desc-sub-{subject}_mask.nii.gz"
    )
    nib.save(subject_mask_nii, subject_mask_path)

    return subject_mask_nii, subject_mask_path


def make_subject_parcel(
    subject: str,
    subject_mask_nii: nib.nifti1.Nifti1Image,
    sp: Study_Params,
) -> (nib.nifti1.Nifti1Image, str):
    """.

    Resample parcellation atlas to subject grey matter mask.
    """
    parcellation = nib.load(sp.parcellation_path)
    subject_parcel_nii = resample_to_img(
        parcellation, subject_mask_nii, interpolation="nearest"
    )
    subject_parcel_path = (
        f"{sp.mask_dir}/tpl-{sp.template}_"
        f"atlas-{sp.atlas_name}_"
        "res-dataset_"
        f"desc-{sp.desc}_"
        f"{sp.atlas_type}_sub-{subject}.nii.gz"
    )
    nib.save(subject_parcel_nii, subject_parcel_path)

    return subject_parcel_nii, subject_parcel_path


def prep_subject(
    subject: str,
    subject_parcel_path: str,
    sp: Study_Params,
) -> (NiftiLabelsMasker, str, list):
    """.

    Prepare subject-specific params.
    """
    atlas_path = Path(subject_parcel_path)

    if sp.atlas_type == "dseg":
        atlas_masker = NiftiLabelsMasker(
            labels_img=atlas_path, standardize=False
        )
    elif sp.atlas_type == "probseg":
        atlas_masker = NiftiMapsMasker(maps_img=atlas_path, standardize=False)

    subj_tseries_path = (
        f"{sp.timeseries_dir}/"
        f"sub-{subject}_friends_BOLDtimeseries_atlas-"
        f"{sp.atlas_name}{sp.desc}"
        f"_desc-{sp.strategy["name"]}.h5"
    )

    if Path(subj_tseries_path).exists():
        with h5py.File(subj_tseries_path, 'r') as f:
            processed_episode_list = [f for f in f.keys()]
    else:
        processed_episode_list = []

    return atlas_masker, subj_tseries_path, processed_episode_list


def get_tseries(
    img: str,
    subject_mask_path: str,
    atlas_masker: NiftiLabelsMasker,
    sp: Study_Params,
) -> (np.array, np.array):
    """.

    Extract timeseries and connectome from denoised volume.
    """
    denoised_img = denoise_nifti_voxel(
        sp.strategy,
        subject_mask_path,
        sp.standardize,
        sp.args.smoothing_fwhm,
        img,
    )

    if not denoised_img:
        print(f"{img} : no volume after scrubbing")
        return None, None

    else:
        (
            correlation_matrix,
            time_series,
        ) = generate_timeseries_connectomes(
            atlas_masker,
            denoised_img,
            subject_mask_path,
            sp.correlation_measure,
            sp.calcul_avgcorr,
        )

        # hack: remove last TR from run that stopped one run short for sub-06
        if f"{task.split('-')[-1]}" == 's01e12a':
            time_series = time_series[:471, :]

        return time_series, correlation_matrix


def save_tseries(
    subj_tseries_path: str,
    task: str,
    time_series: np.array,
    comp_args: dict,
) -> None:
    """.

    Save episode's time series in .h5 file.
    """
    flag =  "a" if Path(subj_tseries_path).exists() else "w"
    with h5py.File(subj_tseries_path, flag) as f:
        if not f"{task.split('-')[-1]}" in f.keys():
            group = f.create_group(f"{task.split('-')[-1]}")

            timeseries_dset = group.create_dataset(
                "timeseries",
                data=time_series,
                **comp_args,
            )
            timeseries_dset.attrs["RepetitionTime"] = 1.49


def extract_subject_timeseries(
    bold_list: list,
    processed_episodes: list,
    subject_mask_path: str,
    atlas_masker: NiftiLabelsMasker,
    subj_tseries_path: str,
    sp: Study_Params,
) -> None:
    """.

    Generate subject's run-level time series.
    """
    for img in tqdm(bold_list):
        try:
            # parse file name
            sub, ses, task, space, _, _ = img.split('/')[-1].split('_')
            print(sub, task)

            if not f"{task.split('-')[-1]}" in processed_episodes:
                # process timeseries
                time_series, _ = get_tseries(
                    img,
                    subject_mask_path,
                    atlas_masker,
                    sp,
                )

                if time_series is not None:
                    save_tseries(
                        subj_tseries_path,
                        task,
                        time_series,
                        sp.comp_args,
                    )

        except:
            print(f"could not process file {img}" )

    return


def extract_timeseries(
    args: argparse.Namespace,
) -> None:
    """.

    Extract time series from atlas parcels.
    """

    print(vars(args))

    # set up path and analysis parameters
    sp = Study_Params(args, **STUDY_PARAMETERS)

    for subject in sp.subjects:
        bold_list, mask_list = compile_bold_list(subject, sp)

        subject_mask_nii, subject_mask_path = make_subjectGM_mask(
            subject,
            mask_list,
            sp,
        )
        subject_parcel_nii, subject_parcel_path = make_subject_parcel(
            subject,
            subject_mask_nii,
            sp,
        )
        atlas_masker, subj_tseries_path, processed_episodes = prep_subject(
            subject,
            subject_parcel_path,
            sp,
        )
        extract_subject_timeseries(
            bold_list,
            processed_episodes,
            subject_mask_path,
            atlas_masker,
            subj_tseries_path,
            sp,
        )


if __name__ == "__main__":
    """
    Process subject-specific fMRIPrep outputs into denoised timeseries
    using MIST parcellation atlas.
    """
    args = get_arguments()
    print(vars(args))

    extract_timeseries(args)
