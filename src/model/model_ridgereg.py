import argparse
import glob
import json
import sys
from pathlib import Path

import h5py
import nibabel as nib
from nilearn.maskers import NiftiLabelsMasker
import numpy as np
from scipy import spatial
from scipy.spatial.distance import cosine
import scipy.stats as stats
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold


def get_arguments() -> argparse.Namespace:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Extracts downsampled, resized frames from movie files (.mkv)"
            " and exports them as .h5 files."
        ),
    )
    parser.add_argument(
        "--idir",
        type=str,
        required=True,
        help="Path to directory with input data.",
    )
    parser.add_argument(
        "--bdir",
        type=str,
        required=True,
        help="Path to directory with BOLD timeseries data.",
    )
    parser.add_argument(
        "--odir",
        type=str,
        required=True,
        help="Path to output directory where season-specific"
        " .h5 frame files will be saved.",
    )
    parser.add_argument(
        "--adir",
        type=str,
        default=None,
        help="Path to directory with subject-specific parcelated"
        " atlases used to extract BOLD timeseries.",
    )
    parser.add_argument(
        "--participant",
        type=str,
        required=True,
        help="CNeuroMod participant label. E.g., sub-01.",
    )
    parser.add_argument(
        "--modalities",
        choices=['visual', 'audio', 'text'],
        help="Input modalities to include in ridge regression.",
        nargs="+",
    )
    parser.add_argument(
        "--back",
        type=int,
        default=5,
        choices=range(0, 10),
        help="How far back in time (in TRs) does the input window start "
        "in relation to the TR it predicts. E.g., back = 5 means that input "
        "features are sampled starting 5 TRs before the target BOLD TR onset",
    )
    parser.add_argument(
        "--input_duration",
        type=int,
        default=3,
        choices=range(1, 5),
        help="Duration of input time window (in TRs) to predict a BOLD TR. "
        "E.g., input_duration = 3 means that input is sampled over 3 TRs "
        "to predict a target BOLD TR.",
    )
    parser.add_argument(
        "--n_split",
        type=int,
        default=8,
        help="Number of cross-validation folds.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=None,
        help="Set seed to assign runs to train & validation sets.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        help="Set to 1 for extra information. Default is 0.",
    )

    return parser.parse_args()


def split_episodes(
    bold_dir: str,
    subject_id: str,
    n_split: int,
    random_state: int=None,
) -> tuple:
    """.

    Assigns subject's runs to train, validation and test sets
    """
    sub_h5 = h5py.File(
                f"{bold_dir}/"
                f"{subject_id}_friends_BOLDtimeseries_"
                "atlas-MIST444_desc-simple+gsr.h5",
                "r",
                )

    # Season 3 held out for test set
    test_set = [x for x in sub_h5 if x[:3] == 's03']

    # Remaining runs assigned to train and validation sets
    r = np.random.RandomState(random_state)  # select season for validation set

    if subject_id == 'sub-04':
        val_season = r.choice(["s01", "s02", "s04"], 1)[0]
    else:
        val_season = r.choice(["s01", "s02", "s04", "s05", "s06"], 1)[0]
    val_set = [x for x in sub_h5 if x[:3] == val_season]
    train_set = sorted([x for x in sub_h5 if x[:3] not in ['s03', val_season]])

    sub_h5.close()

    # Assign consecutive train set episodes to cross-validation groups
    lts = len(train_set)
    train_groups = np.floor(np.arange(lts)/(lts/n_split)).astype(int).tolist()

    return train_groups, train_set, val_set, test_set


def build_audio_visual(
    idir: str,
    modalities: list,
    runs: list,
    run_lengths: list,
    duration: int,
) -> np.array:
    """.

    Concatenates visual and audio features into array.
    """

    x_list = []

    for run, rl in zip(runs, run_lengths):
        season: str = run[2]

        h5_path = Path(
            f"{idir}/friends_s{season}_"
            "features_visual_audio_gzip_level-4.h5"
        )

        run_input = {}
        with h5py.File(h5_path, "r") as f:
            for modality in modalities:
                run_input[modality] = np.array(f[run][modality])

        run_list = []

        for modality in modalities:
            for k in range(duration):
                run_list.append(
                    np.nan_to_num(
                        # default computes zscore over axis 0
                        stats.zscore(
                            run_input[modality][k:(rl+k), :]
                        )
                    )
                )

        x_list.append(np.concatenate(run_list, axis=1))

    return np.concatenate(x_list, axis=0)


def build_text(
    idir: str,
    runs: list,
    run_lengths: list,
    duration: int,
) -> np.array:

    dur = duration - 1
    feature_list = ['text_pooled', 'text_token']
    x_dict = {
        "text_pooled": [],
        "text_token": [],
    }

    for run, rl in zip(runs, run_lengths):
        season: str = run[2]

        h5_path = Path(
            f"{idir}/friends_s{season}_"
            "features_text_gzip_level-4.h5"
        )

        with h5py.File(h5_path, "r") as f:
            for feat_type in feature_list:
                run_data = np.array(f[run][feat_type])[dur: dur+rl, :]

                # pad features array in case fewer text TRs than for BOLD data
                rdims = run_data.shape

                rsize = rl*rdims[1] if len(rdims) == 2 else rl*rdims[1]*rdims[2]
                run_array = np.repeat(np.nan, rsize).reshape((rl,) + rdims[1:])
                run_array[:rdims[0]] = run_data

                x_dict[feat_type].append(run_array)

    x_list = []
    for feat_type in feature_list:
        feat_data = np.concatenate(x_dict[feat_type], axis=0)
        dims = feat_data.shape

        x_list.append(
            np.nan_to_num(
                stats.zscore(
                    feat_data.reshape((-1, dims[-1])),
                    nan_policy="omit",
                    axis=0,
                )
            ).reshape(dims).reshape(dims[0], -1).astype('float32')
        )

    return np.concatenate(x_list, axis=1)


def build_input(
    idir: str,
    modalities: list,
    runs: list,
    run_lengths: list,
    duration: int,
) -> np.array:
    """.

    Concatenates input features across modalities into predictor array.
    """

    x_list = []

    av_modalities = [x for x in modalities if x in ["vision", "audio"]]
    if len(av_modalities) > 0:
        x_list.append(
            build_audio_visual(
                idir,
                av_modalities,
                runs,
                run_lengths,
                duration,
            ),
        )

    if "text" in modalities:
        x_list.append(
            build_text(
                idir,
                runs,
                run_lengths,
                duration,
            ),
        )

    if len(x_list) > 1:
        return np.concatenate(x_list, axis=1)
    else:
        return x_list[0]


def build_output(
    bold_dir: str,
    subject_id: str,
    runs: list,
    start: int,
    run_groups: list = None,
) -> tuple:
    """.

    Concatenates BOLD timeseries into target array.
    """
    y_list = []
    length_list = []
    y_groups = []
    sub_h5 = h5py.File(
                f"{bold_dir}/"
                f"{subject_id}_friends_BOLDtimeseries_"
                "atlas-MIST444_desc-simple+gsr.h5",
                "r",
                )

    for i, run in enumerate(runs):
        run_ts = np.array(sub_h5[run]['timeseries'])[start:, :]
        length_list.append(run_ts.shape[0])
        y_list.append(run_ts)

        if run_groups is not None:
            y_groups.append(np.repeat(run_groups[i], run_ts.shape[0]))

    sub_h5.close()
    y_list = np.concatenate(y_list, axis=0)
    y_groups = np.concatenate(y_groups, axis=0) if run_groups is not None else np.array([])

    return y_list, length_list, y_groups


def train_ridgeReg(
    X: np.array,
    y: np.array,
    groups: list,
    n_splits: int,
) -> RidgeCV:
    """.

    Performs ridge regression with built-in cross-validation.
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html
    """
    alphas = np.logspace(0.1, 3, 10)
    group_kfold = GroupKFold(n_splits=n_splits)
    cv = group_kfold.split(X, y, groups)

    model = RidgeCV(
        alphas=alphas,
        fit_intercept=True,
        #normalize=False,
        cv=cv,
    )

    return model.fit(X, y)


def pairwise_acc(
    target: np.array,
    predicted: np.array,
    use_distance: bool = False,
) -> float:
    """.

    Computes Pairwise accuracy
    Adapted from: https://github.com/jashna14/DL4Brain/blob/master/src/evaluate.py
    """
    true_count = 0
    total = 0

    for i in range(0,len(target)):
        for j in range(i+1, len(target)):
            total += 1

            t1 = target[i]
            t2 = target[j]
            p1 = predicted[i]
            p2 = predicted[j]

            if use_distance:
                if cosine(t1,p1) + cosine(t2,p2) < cosine(t1,p2) + cosine(t2,p1):
                    true_count += 1

            else:
                if pearsonr(t1,p1)[0] + pearsonr(t2,p2)[0] > pearsonr(t1,p2)[0] + pearsonr(t2,p1)[0]:
                    true_count += 1

    return (true/total)


def pearson_corr(
    target: np.array,
    predicted: np.array,
) -> np.array:
    """.

    Calculates pearson R between predictions and targets.
    """
    r_vals = []
    for i in range(len(target)):
        r_val, _  = pearsonr(target[i], predicted[i])
        r_vals.append(r_val)

    return np.array(r_vals)


def export_images(
    odir: str,
    adir: str,
    subject: str,
    modal_names: str,
    results: dict,
) -> None:
    """.

    Exports RR parcelwise scores as nifti files with
    subject-specific atlas used to extract timeseries.
    """
    atlas_path = Path(
        f"{adir}/tpl-MNI152NLin2009cAsym_atlas-MIST_"
        f"res-dataset_desc-444_dseg_{subject}.nii.gz"
    )
    atlas_masker = NiftiLabelsMasker(
        labels_img=atlas_path,
        standardize=False,
    )
    atlas_masker.fit()

    # map Pearson correlations onto brain parcels
    for s in ['train', 'val']:
        nii_file = atlas_masker.inverse_transform(
            np.array(results["parcelwise"][f"{s}_R2"]),
        )
        nib.save(
            nii_file,
            f"{odir}/{subject}_MIST444_RidgeReg{modal_names}_R2_{s}.nii.gz",
        )

    return


def test_ridgeReg(
    odir: str,
    subject: str,
    R: RidgeCV,
    x_train: np.array,
    y_train: np.array,
    x_val: np.array,
    y_val: np.array,
    modalities: list,
    adir: str=None,
) -> None:
    """.

    Exports RR results in .json file.
    """
    res_dict = {}

    # Global R2 score
    res_dict["train_R2"] = R.score(x_train, y_train)
    res_dict["val_R2"] = R.score(x_val, y_val)

    # Parcel-wise predictions
    pred_train = R.predict(x_train)
    pred_val = R.predict(x_val)

    res_dict["parcelwise"] = {}
    res_dict["parcelwise"]["train_R2"] = (
                                    pearson_corr(
                                        y_train.T,
                                        pred_train.T
                                        )**2
                                    ).tolist()
    res_dict["parcelwise"]["val_R2"] = (
                                    pearson_corr(
                                    y_val.T,
                                    pred_val.T
                                    )**2
                                    ).tolist()

    # export RR results
    Path(f"{odir}").mkdir(parents=True, exist_ok=True)
    m = ""
    for modal in modalities:
        m += f"_{modal}"
    with open(f"{odir}/{subject}_ridgeReg{m}_result.json", 'w') as fp:
        json.dump(res_dict, fp)

    # export parcelwise scores as .nii images for visualization
    if adir is not None:
        export_images(odir, adir, subject, m, res_dict)


def main() -> None:
    """.

    This script performs a ridge regression on the Courtois Neuromod friends
    dataset.
    It uses multimodal features (visual / audio / text) extracted from the
    videos to predict parcellated BOLD time series.
    """
    args = get_arguments()

    print(vars(args))

    # Assign runs to train / validation / test sets
    train_grps, train_runs, val_runs, test_runs = split_episodes(
        args.bdir,
        args.participant,
        args.n_split,
        args.random_state,
    )

    # Build y matrix from BOLD timeseries
    y_train, length_train, train_groups = build_output(
        args.bdir,
        args.participant,
        train_runs,
        args.back,
        train_grps,
    )
    y_val, length_val, _ = build_output(
        args.bdir,
        args.participant,
        val_runs,
        args.back,
    )

    # Build X arrays from input features
    x_train = build_input(
        args.idir,
        args.modalities,
        train_runs,
        length_train,
        args.input_duration,
    )
    x_val = build_input(
        args.idir,
        args.modalities,
        val_runs,
        length_val,
        args.input_duration,
    )

    # Train ridge regression model on train set
    model = train_ridgeReg(
        x_train,
        y_train,
        train_groups,
        args.n_split,
    )

    # Test model and export performance metrics
    test_ridgeReg(
        args.odir,
        args.participant,
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        args.modalities,
        args.adir,
    )


if __name__ == "__main__":
    sys.exit(main())
