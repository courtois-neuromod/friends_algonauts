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
        "--seasons",
        default=None,
        nargs="+",
        help="List of seasons of .mkv files to process. If none is specified, "
        "all seasons will be processed.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="Data type for final arrays (chunks of resized, "
        "downsampled movie frames)",
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


def set_target_dims(
    strategy: str,
    target_height: int,
    target_width: int = None,
) -> (str, int, int, int, int):
    """Validates resizing strategy and specified dims.

    Corrects strategy and dims if necessary.
    Returns the intermediate (after resizing) and
    final (as exported) frame height and width,
    in pixels, according to specified resizing strategy.

    Args:
        strategy (str): user-specified resizing strategy.
        target_height (int): final target heigh in pixels (required).
        target_width (int): final target width in pixels. Default=None.

    Returns: (
        final_strategy (str): final resizing strategy,
        intermediate_height (int): frame height after resizing,
        intermediate_width (int): frame width after resizing,
        final_height (int): saved frame height,
        final_width (int): saved frame width,
    )
    """
    if strategy != "custom" and target_width is not None:
        print(
            "Warning: frames will be resized to user-specified dimensions "
            f"(height = {target_height}; width = {target_width}).",
        )
        print(
            f'Resize strategy changed from "{strategy.upper()}" to "CUSTOM". '
            "Final array height may not match width!",
        )
        final_strategy = "custom"
    elif strategy == "custom" and target_width is None:
        print(
            "Warning: a target width needs to be specified to use the "
            '"custom" resize strategy.',
        )
        print(
            f'Resize strategy changed from "{strategy.upper()}" to '
            '"PROPORTIONAL". Final array height =/= width!',
        )
        final_strategy = "proportional"
    else:
        final_strategy = strategy

    """
    Note that intermediate height and width reflect the frame
    dimensions after the image is resized.

    Final height and width reflect the saved frame dimensions,
    following additional steps like cropping or padding.

    Original frame dims = (height: 480, weidth: 720) pixels.
    """
    if final_strategy == "custom":
        """
        The CUSTOM resize strategy resizes the height and width
        to user-specified dimensions.
        """
        intermediate_height = target_height
        intermediate_width = target_width

        final_height = intermediate_height
        final_width = intermediate_width

    elif final_strategy == "proportional":
        """
        The PROPORTIONAL resize strategy adjusts the target width
        to the target height to maintain original frame proportions.
        Output frames (arrays) are rectangular (height =/= width).
        """
        intermediate_height = target_height
        intermediate_width = int((target_height * 720) / 480)

        final_height = intermediate_height
        final_width = intermediate_width

    elif final_strategy == "squeeze":
        """
        The SQUEEZE resize strategies transform frames into
        squared arrays (height == width). It introduces distortion
        (squeezing along width) but no feature loss.
        """
        intermediate_height = target_height
        intermediate_width = target_height

        final_height = intermediate_height
        final_width = intermediate_width

    elif final_strategy == "crop":
        """
        The CROP resize strategies transform frames into
        squared arrays (height == width). It introduces no distortion
        but some feature loss on the left and right of the frame.
        """
        intermediate_height = target_height
        intermediate_width = int((target_height * 720) / 480)

        final_height = intermediate_height
        final_width = final_height

    elif final_strategy == "pad":
        """
        The PAD resize strategies transform frames into
        squared arrays (height == width). It involves no distortion
        and no feature loss, but requires additional storage for
        empty pixels added to the top and bottom of the frame.
        """
        intermediate_height = int((target_height * 480) / 720)
        intermediate_width = target_height

        final_height = target_height
        final_width = target_height

    return (
        final_strategy,
        intermediate_height,
        intermediate_width,
        final_height,
        final_width,
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

    print(vars(args))
    (
        strategy,
        rs_height,
        rs_width,
        target_height,
        target_width,
    ) = set_target_dims(
        args.resize_strategy,
        args.target_height,
        args.target_width,
    )
    print(
        f"\nResize Strategy: {strategy.upper()} \nResized frame dimensions in "
        f"pixels: h = {target_height}, w = {target_width}.\n",
    )

    n_frames_float = STUDY_PARAMS["tr"] * STUDY_PARAMS["fps"]
    n_frames = int(n_frames_float)
    print(
        "Downsampled frames-per-second rate: "
        f"{STUDY_PARAMS['fps']/ args.time_downsample} (orig = 29.97 fps)",
    )
    print(
        f"Final arrays include {int(n_frames/args.time_downsample)} frames "
        "per chunk of 1.49s (one fMRI TR).\n",
    )

    seasons = (
        args.seasons
        if args.seasons is not None
        else [x.split("/")[-1] for x in sorted(glob.glob(f"{args.idir}/s*"))]
    )
    print("Seasons : ", seasons)

    for season in seasons:
        episode_list = [
            x.split("/")[-1].split(".")[0][-7:]
            for x in sorted(glob.glob(f"{args.idir}/{season}/friends_s*.mkv"))
            if x.split("/")[-1].split(".")[0][-1] in ["a", "b", "c", "d"]
        ]
        if args.verbose:
            print(episode_list)

        """
        Set .h5 array compression parameters
        Doc here: https://docs.h5py.org/en/stable/high/dataset.html#filter-pipeline
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
            f"{args.odir}/friends_{season}_frames_h-{target_height}_"
            f"w-{target_width}_{strategy.upper()}{pv}_ds-{args.time_downsample}"
            f"{compress_details}.h5"
        )

        """
        To re-launch an interrupted script
        """
        processed_episodes = []
        if Path(out_file).exists():
            season_h5_file = h5py.File(out_file, "r")
            processed_episodes = list(season_h5_file.keys())
            if args.verbose:
                print("Processed episodes : ", processed_episodes)
            season_h5_file.close()

        for episode in tqdm(episode_list, desc="processing .mkv files"):
            if episode not in processed_episodes:
                mkv_path = f"{args.idir}/{season}/friends_{episode}.mkv"
                if Path(mkv_path).exists():
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
                            # temporal downsampling, rescale [-0.5, 0.5]
                            frames = (
                                np.asarray(
                                    chunk_frames[:n_frames],
                                    dtype=args.dtype,
                                )[:: args.time_downsample]
                                / 255.0
                                - 0.5
                            )

                            """pytorch / chainer input dimension order:
                            Channel x Frame x Height x Width
                            F, H, W, C -> C, F, H, W
                            """
                            chunk_array = np.transpose(
                                frames,
                                [3, 0, 1, 2],
                            )
                            # must be color channel
                            if not chunk_array.shape[0] == 3:
                                raise ValueError
                            if not chunk_array.shape[1] == np.ceil(
                                n_frames / args.time_downsample,
                            ):
                                raise ValueError
                            if not chunk_array.shape[2] == rs_height:
                                raise ValueError
                            if not chunk_array.shape[3] == rs_width:
                                raise ValueError

                            if strategy == "crop":
                                edge = int(
                                    np.floor((rs_width - target_width) / 2),
                                )
                                chunk_array = chunk_array[
                                    :,
                                    :,
                                    :,
                                    edge : target_width + edge,
                                ]

                            elif strategy == "pad":
                                padvox_val = (
                                    args.padvox_intensity / 255.0
                                ) - 0.5
                                full_array = np.full(
                                    (
                                        3,
                                        int(n_frames / args.time_downsample),
                                        target_height,
                                        target_width,
                                    ),
                                    padvox_val,
                                    dtype=args.dtype,
                                )
                                edge = int(
                                    np.floor((target_height - rs_height) / 2),
                                )
                                full_array[
                                    :,
                                    :,
                                    edge : rs_height + edge,
                                    :,
                                ] = chunk_array
                                chunk_array = full_array

                            if not chunk_array.shape[0] == 3:
                                raise ValueError
                            if not chunk_array.shape[1] == np.ceil(
                                n_frames / args.time_downsample,
                            ):
                                raise ValueError
                            if not chunk_array.shape[2] == target_height:
                                raise ValueError
                            if not chunk_array.shape[3] == target_width:
                                raise ValueError

                            chunk_dict[chunk_count] = chunk_array

                            chunk_count += 1

                    flag = "a" if Path(out_file).exists() else "w"
                    with h5py.File(out_file, flag) as f:
                        group = f.create_group(episode)

                        for c in chunk_dict:
                            group.create_dataset(
                                f"{str(c).zfill(3)}",
                                data=chunk_dict[c],
                                **comp_args,
                            )

                    cap.release()


if __name__ == "__main__":
    sys.exit(main())
