import argparse
import glob
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
from skimage.transform import resize
from tqdm import tqdm

STUDY_PARAMS = {
    "fps": 29.97,
    "tr": 1.49,
}


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
        help="Path to input directory that contains sub-directories"
        " (s1, s2, etc.) with .mkv files organized per season.",
    )
    parser.add_argument(
        "--odir",
        type=str,
        required=True,
        help="Path to output directory where season-specific"
        " .h5 frame files will be saved.",
    )
    parser.add_argument(
        "--time_downsample",
        type=int,
        default=2,
        choices=range(1, 45),
        metavar="[1-44]",
        help="Temoral downsampling factor (1, 45). "
        "1 = no downsampling. 3 = keeps 1 out of 3 frames.",
    )
    parser.add_argument(
        "--target_height",
        type=int,
        default=128,
        help="Dimension to which frame height is resized, in pixels",
    )
    parser.add_argument(
        "--target_width",
        type=int,
        default=None,
        help="Dimension to which frame width is resized, in pixels. "
        'Only required for "custom" resizing strategy.',
    )
    parser.add_argument(
        "--resize_strategy",
        type=str,
        default="proportional",
        choices=["proportional", "squeeze", "crop", "pad", "custom"],
        help="Strategy used to resize movie frames.",
    )
    parser.add_argument(
        "--padvox_intensity",
        type=int,
        default=0,
        choices=range(0, 256),
        help="Padding pixel intensity value, from 0-255. Only used for "
        '"Padding" resizing strategy',
    )
    parser.add_argument(
        "--normalize",
        action='store_true',
        help="if True, normalizes pixel values (0 centered, range = [-0.5, 0.5]). "
        "Default is False.",
    )
    parser.add_argument(
        "--seasons",
        default=None,
        nargs="+",
        help="List of seasons of .mkv files to process. If none is specified, "
        "all seasons will be processed.",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default=None,
        choices=[None, "gzip", "lzf"],
        help="Compression to apply to frames. Default is none.",
    )
    parser.add_argument(
        "--compression_opts",
        type=int,
        default=4,
        choices=range(0, 10),
        help="Frame compression level in .h5 file. Value = [0-9]. "
        "Only for lossless gzip compression.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        help="Set to 1 for extra information. Default is 0.",
    )

    return parser.parse_args()


def validate_strategy(
    strategy: str,
    t_w: int,
) -> None:
    """.

    Validate choice of strategy.
    """
    if strategy == "custom":
        if t_w is None:
            raise Exception(
                "A target width needs to be specified with the CUSTOM strategy."
            )
    else:
        if t_w is not None:
            raise Exception(
                "The specified target width is incompatible with the "
                f"{strategy.upper()} strategy."
            )

    return


def set_target_dims(
    args,
) -> (str, tuple, tuple):
    """.

    Get post-resize and final frame dims.
    Original frame dims: (h=480, w=720 pixels).
    """
    validate_strategy(
        args.resize_strategy,
        args.target_width,
    )

    if args.resize_strategy == "custom":
        return (
            args.resize_strategy,
            (args.target_height, args.target_width),
            (args.target_height, args.target_width),
        )

    elif args.resize_strategy == "proportional":
        return (
            args.resize_strategy,
            (args.target_height, int((args.target_height * 720) / 480)),
            (args.target_height, int((args.target_height * 720) / 480)),
        )

    elif args.resize_strategy == "squeeze":
        return (
            args.resize_strategy,
            (args.target_height, args.target_height),
            (args.target_height, args.target_height),
        )

    elif args.resize_strategy == "crop":
        return (
            args.resize_strategy,
            (args.target_height, int((args.target_height * 720) / 480)),
            (args.target_height, args.target_height),
        )

    elif args.resize_strategy == "pad":
        return (
            args.resize_strategy,
            (int((args.target_height * 480) / 720), args.target_height),
            (args.target_height, args.target_height),
        )
    else:
        raise Exception()


def list_seasons(
    idir: str,
    verbose: int,
    seasons: list=None,
) -> list:
    """.

    List of seasons to process.
    """
    season_list = (
        seasons if seasons is not None
        else [x.split("/")[-1] for x in sorted(glob.glob(f"{idir}/s*"))]
    )
    if verbose:
        print("Seasons : ", seasons)

    return season_list


def set_output(
    season: str,
    strategy: str,
    target_dim: tuple,
    args: argparse.Namespace,
) -> tuple:
    """.

    Set output parameters.
    """
    compress_details = ""
    comp_args = {}
    if args.compression is not None:
        compress_details = f"_{args.compression}"
        comp_args["compression"] = args.compression
        if args.compression == "gzip":
            compress_details += f"_level-{args.compression_opts}"
            comp_args["compression_opts"] = args.compression_opts

    pv = f"_padval-{args.padvox_intensity}" if strategy == "pad" else ""
    out_file = (
        f"{args.odir}/friends_{season}_frames_h-{target_dim[0]}_"
        f"w-{target_dim[1]}_{strategy.upper()}{pv}_ds-{args.time_downsample}"
        f"{compress_details}.h5"
    )

    Path(f"{args.odir}").mkdir(exist_ok=True, parents=True)
    dtype = "float32" if args.normalize else "uint8"

    return dtype, comp_args, out_file


def list_episodes(
    idir: str,
    season: str,
    outfile: str,
    verbose: int,
) -> list:
    """.

    Compile season's list of episodes to process.
    """
    all_epi = [
        x.split("/")[-1].split(".")[0][-7:]
        for x in sorted(glob.glob(f"{idir}/{season}/friends_s*.mkv"))
        if x.split("/")[-1].split(".")[0][-1] in ["a", "b", "c", "d"]
    ]

    if Path(outfile).exists():
        season_h5_file = h5py.File(outfile, "r")
        processed_epi = list(season_h5_file.keys())
        season_h5_file.close()
    else:
        processed_epi = []

    episode_list = [epi for epi in all_epi if epi not in processed_epi]

    if verbose:
        print("Processed episodes : ", processed_epi)
        print("Episodes to process : ", episode_list)

    return episode_list


def pad_array(
    chunk_array: np.array,
    n_frames: int,
    rs_dim: tuple,
    target_dim: tuple,
    dtype: str,
    args: argparse.Namespace,
) -> np.array:
    """.

    """
    rs_height, rs_width = rs_dim
    target_height, target_width = target_dim

    padvox_val = args.padvox_intensity
    if args.normalize:
        padvox_val = (padvox_val / 255.0) - 0.5

    full_array = np.full(
        (
            3,
            int(np.ceil(n_frames / args.time_downsample)),
            target_height,
            target_width,
        ),
        padvox_val,
        dtype=dtype,
    )

    edge = int(np.floor((target_height - rs_height) / 2))
    full_array[:, :, edge : rs_height + edge, :, ] = chunk_array

    return full_array


def make_array(
    chunk_frames: list,
    strategy: str,
    dtype: str,
    rs_dim: tuple,
    target_dim: tuple,
    args: argparse.Namespace,
) -> np.array:
    """.

    Extract frames from an episode, in chunks of duration = fMRI TR.
    """
    n_frames = int(STUDY_PARAMS["tr"] * STUDY_PARAMS["fps"])
    rs_height, rs_width = rs_dim
    target_height, target_width = target_dim

    # temporal downsampling, rescale [-0.5, 0.5]
    frames = (
        np.asarray(
            chunk_frames[:n_frames],
            dtype=dtype,
        )[:: args.time_downsample]
    )

    if args.normalize:
        frames = (frames / 255.0) - 0.5

    """pytorch / chainer input dimension order:
    Channel x Frame x Height x Width
    F, H, W, C -> C, F, H, W
    """
    chunk_array = np.transpose(
        frames,
        [3, 0, 1, 2],
    )

    if not chunk_array.shape == (
        3,
        int(np.ceil(n_frames / args.time_downsample)),
        rs_height,
        rs_width,
    ):
        raise ValueError

    if strategy == "crop":
        edge = int(np.floor((rs_width - target_width) / 2))
        chunk_array = chunk_array[:, :, :, edge : target_width + edge]

    elif strategy == "pad":
        chunk_array = pad_array(
            chunk_array,
            n_frames,
            rs_dim,
            target_dim,
            dtype,
            args,
        )

    if not chunk_array.shape == (
        3,
        int(np.ceil(n_frames / args.time_downsample)),
        target_height,
        target_width,
    ):
        raise ValueError

    return chunk_array


def save_chunks(
    outfile_name: str,
    episode: str,
    chunk_dict: dict,
    comp_args: dict,
) -> None:
    """.

    Exports episode's chunks of frames as numbered arrays in .h5 file.
    """
    flag = "a" if Path(outfile_name).exists() else "w"
    with h5py.File(outfile_name, flag) as f:
        group = f.create_group(episode)

        for c in chunk_dict:
            group.create_dataset(
                f"{str(c).zfill(3)}",
                data=chunk_dict[c],
                **comp_args,
            )


def extract_features(
    season: str,
    episode: str,
    mkv_path: str,
    strategy: str,
    rs_dim: tuple,
    target_dim: tuple,
    dtype: int,
    outfile_name: str,
    comp_args: dict,
    args: argparse.Namespace,
) -> None:
    """.

    Extract frames from an episode, in chunks of duration = 1 fMRI TR.
    """
    n_frames_float = STUDY_PARAMS["tr"] * STUDY_PARAMS["fps"]
    rs_height, rs_width = rs_dim

    cap = cv2.VideoCapture(mkv_path)
    if args.verbose:
        print(
            f"Episode {episode} frame count: "
            f"{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}",
        )

    chunk_dict = {}
    chunk_count = 0
    frame_count = 0
    success = True

    while success:
        chunk_frames = []
        if args.verbose:
            print(chunk_count, frame_count)

        while frame_count < int((chunk_count + 1) * n_frames_float):
            success, image = cap.read()

            if success:
                # flip color channels to be RGB (cv2)
                chunk_frames.append(
                    np.floor(
                        resize(
                            image[..., ::-1],
                            (rs_height, rs_width),
                            preserve_range=True,
                            anti_aliasing=True,
                        ),
                    ).astype("uint8"),
                )
            frame_count += 1

        # only process complete chunks
        if success:
            chunk_dict[chunk_count] = make_array(
                chunk_frames,
                strategy,
                dtype,
                rs_dim,
                target_dim,
                args,
            )
            chunk_count += 1

    cap.release()
    save_chunks(outfile_name, episode, chunk_dict, comp_args)


def process_episodes(
    season: str,
    strategy: str,
    rs_dim: tuple,
    target_dim: tuple,
    args: argparse.Namespace,
) -> None:
    """.

    Extract frames from season's episodes.
    """
    # set output params
    dtype, comp_args, outfile_name = set_output(
        season,
        strategy,
        target_dim,
        args,
    )

    episode_list = list_episodes(args.idir, season, outfile_name, args.verbose)

    for episode in tqdm(episode_list, desc="processing .mkv files"):
        mkv_path = f"{args.idir}/{season}/friends_{episode}.mkv"

        if Path(mkv_path).exists():
            extract_features(
                season,
                episode,
                mkv_path,
                strategy,
                rs_dim,
                target_dim,
                dtype,
                outfile_name,
                comp_args,
                args,
            )


def main() -> None:
    """.

    This script resizes and downsamples movie frames into arrays
    and stores them into .h5 files. The inputs are .mkv files that
    correspond to episodes of the sitcom Friends, split into ~12 min
    segments that were shown to participants during fMRI runs.
    The majority of episodes are chunked into two segments (a and b).
    A handful of special "double-episodes" are chunked into segments
    a, b, c and d.

    Frames are saved into chunks that correspond to 1.49s of movie
    watching. This duration matches the temporal frequency at which
    fMRI brain volumes were acquired for the current dataset
    (TR = 1.49s), to facilitate the pairing between movie frames
    and brain data.

    One .h5 output file is generated per season. Within each file,
    chunks of frames are organized hierarchically per episode segment
    (e.g., s06e03a, the first segment of episode 3 from season 6),
    and then temporally (chunk numbers reflect their order of occurent
    within an episode segment/fMRI run).

    The script resizes images to specified dimensions according to one
    of five strategies:
    - CUSTOM: user specifies height and width.
        Frames are resized to match the specified dimensions.
        No feature loss. Distortion depends on dimensions,
        output may be rectangular.
    - PROPORTIONAL: user specifies height.
        Width is adjusted to maintain original frame proportions.
        No feature loss, no distortion, output is rectangular.
    - SQUEEZE: user specifies height.
        Frame is squeezed along width to match height.
        No feature loss, output is square.
        Not appropriate for pre-trained vision models.
    - CROP: user specifies height.
        Frame is cropped on both sides for width and height to match.
        Output is square, some features lost on the left and right.
    - PAD: user specifies final height.
        Frame is resized proportionally so the width matches
        the specified height, then padded along the height with
        pixels at the top and bottom so height and width match.
        Default padding pixel intensity is black, can be set [0-255].
        No feature loss, output is square, some storage is wasted on
        empty pixels.

    The script can downsample the movie frames to reduce the input
    temporal frequency. The default (time_downsample = 1) maintains
    the movie's original frequency (no frames are dropped).
    A downsampling factor of 3 (time_downsample = 3) indicates
    that one out of 3 frames is kept. The maximal downsampling factor
    is 44, results in 1 kept movie frame per fMRI TR.
    """
    args = get_arguments()

    # get resizing parameters
    strategy, rs_dim, target_dim = set_target_dims(args)

    # Get list of seasons to process
    seasons = list_seasons(args.idir, args.verbose, args.seasons)

    for season in seasons:
        process_episodes(
            season,
            strategy,
            rs_dim,
            target_dim,
            args,
        )


if __name__ == "__main__":
    sys.exit(main())
