import argparse
import glob
import sys
from pathlib import Path

import h5py
import librosa
from moviepy.editor import VideoFileClip
import numpy as np
from tqdm import tqdm

"""
This script relies on the MoviePy library to extract sound files (.wav)
from video files (.mkv).

Moviepy uses FFmpeg under the hood, which may require an installation.
E.g.,
sudo apt update
sudo apt install ffmpeg
"""
STUDY_PARAMS = {
    "sr": 22050,
    "tr": 1.49,
}


def get_arguments() -> argparse.Namespace:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Extracts downsampled soundwaves from movie files (.mkv)"
            " and exports them as .h5 files.",
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
        "--rate_resample",
        type=int,
        default=22050,
        help="Sound wave target resampling rate, in Hz.",
    )
    parser.add_argument(
        "--stereo",
        action="store_true",
        help="if True, export sound wave file as stereo. Default is False (mono).",
    )
    parser.add_argument(
        "--export_wav",
        action='store_true',
        help="if True, saves each .mkv audio as .wav file. Default is False.",
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
    args: argparse.Namespace,
) -> tuple:
    """.

    Set output parameters.
    """
    is_mono = "stereo" if args.stereo else "mono"

    compress_details = ""
    comp_args = {}
    if args.compression is not None:
        compress_details = f"_{args.compression}"
        comp_args["compression"] = args.compression
        if args.compression == "gzip":
            compress_details += f"_level-{args.compression_opts}"
            comp_args["compression_opts"] = args.compression_opts

    out_file = (
        f"{args.odir}/friends_{season}_audio_sr-{args.rate_resample}"
        f"{is_mono}_{compress_details}.h5"
    )

    Path(f"{args.odir}/wav").mkdir(parents=True, exist_ok=True)

    return comp_args, out_file


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


def save_segments(
    outfile_name: str,
    episode: str,
    audio_segments: list,
    comp_args: dict,
) -> None:
    """.

    Exports episode's chunks of soundwaves as numbered arrays in .h5 file.
    """
    flag = "a" if Path(outfile_name).exists() else "w"
    with h5py.File(outfile_name, flag) as f:
        group = f.create_group(episode)

        for i in range(len(audio_segments)):
            group.create_dataset(
                f"{str(i).zfill(3)}",
                data=audio_segments[i],
                **comp_args,
            )


def extract_features(
    season: str,
    episode: str,
    mkv_path: str,
    outfile_name: str,
    comp_args: dict,
    args: argparse.Namespace,
) -> None:
    """.

    Extract soundwaves from an episode, in chunks of duration = 1 fMRI TR.
    """
    n_frames = int(STUDY_PARAMS["tr"] * args.rate_resample)
    mono = not args.stereo

    # extract audio .wav from .mkv
    clip = VideoFileClip(mkv_path)
    if args.export_wav:
        wav_file = f"{args.odir}/wav/friends_{episode}.wav"
    else:
        wav_file = f"{args.odir}/wav/friends_{season}.wav"  # overwrites
    clip.audio.write_audiofile(wav_file)

    audio_segments = []
    for start in np.arange(0, clip.duration, STUDY_PARAMS["tr"]):
        (audio_chunk, _) = librosa.core.load(
            wav_file,
            sr=args.rate_resample,
            mono=mono,
            offset=start,
            duration=STUDY_PARAMS["tr"],
            dtype=args.dtype,
        )
        # only include complete audio chunks
        if not audio_chunk.shape[-1] < n_frames:
            if mono:
                audio_segments.append(audio_chunk[:n_frames])
            else:
                audio_segments.append(audio_chunk[:,:n_frames])

    save_segments(outfile_name, episode, audio_segments, comp_args)


def process_episodes(
    season: str,
    args: argparse.Namespace,
) -> None:
    """.

    Extract soundwaves from season's episodes.
    """
    # set output params
    comp_args, outfile_name = set_output(
        season,
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
                outfile_name,
                comp_args,
                args,
            )

    return


def main() -> None:
    """.

    This script resamples sound waves into arrays
    and stores them into .h5 files. The inputs are .mkv files that
    correspond to episodes of the sitcom Friends, split into ~12 min
    segments that were shown to participants during fMRI runs.
    The majority of episodes are chunked into two segments (a and b).
    A handful of special "double-episodes" are chunked into segments
    a, b, c and d.

    Resampled sound waves are saved into chunks that correspond to 1.49s of
    movie watching. This duration matches the temporal frequency at which
    fMRI brain volumes were acquired for the current dataset
    (TR = 1.49s), to facilitate the pairing between movie frames
    and brain data.

    One .h5 output file is generated per season. Within each file,
    chunks of resampled sound waves are organized hierarchically per episode
    segment (e.g., s06e03a, the first segment of episode 3 from season 6),
    and then temporally (chunk numbers reflect their order of occurent
    within an episode segment/fMRI run).

    The script can resample sound waves to a specified sampling rate (in Hz)
    to modify the input temporal frequency.
    """
    args = get_arguments()
    print(vars(args))

    # Get list of seasons to process
    seasons = list_seasons(args.idir, args.verbose, args.seasons)

    for season in seasons:
        process_episodes(
            season,
            args,
        )


if __name__ == "__main__":
    sys.exit(main())
