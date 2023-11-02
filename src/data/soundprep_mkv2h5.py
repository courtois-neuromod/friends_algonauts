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

https://github.com/brain-bzh/cNeuromod_encoding_2020/blob/refactoring_V2/audio_utils.py
https://github.com/courtois-neuromod/soundnetbrain_hear/blob/main/soundnetbrain_hear/soundnetbrain_hear.py
https://librosa.org/doc/latest/generated/librosa.load.html#librosa.load
"""
STUDY_PARAMS = {
    "sr": 48000,
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

    #n_frames = int(STUDY_PARAMS["tr"] * STUDY_PARAMS["sr"])
    n_frames = int(STUDY_PARAMS["tr"] * args.rate_resample)
    mono = not args.stereo
    is_mono = "stereo" if args.stereo else "mono"
    print(
        "Resampled rate: "
        f"{args.rate_resample} Hz, {is_mono} (orig: 48000 Hz, stereo)",
    )
    print(
        f"Final arrays include {n_frames} frames "
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

        out_file = (
            f"{args.odir}/friends_{season}_audio_sr-{args.rate_resample}"
            f"{is_mono}_{compress_details}.h5",
        )[0]

        Path(f"{args.odir}/wav").mkdir(parents=True, exist_ok=True)
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
                    # extract audio .wav from .mkv
                    clip = VideoFileClip(mkv_path)
                    clip_duration = clip.duration  # movie duration in seconds
                    wav_file = f"{args.odir}/wav/friends_{episode}.wav"
                    clip.audio.write_audiofile(wav_file)

                    audio_segments = []
                    for start in np.arange(0, clip_duration, STUDY_PARAMS["tr"]) :
                        (audio_chunk, _) = librosa.core.load(
                            wav_file,
                            sr=args.rate_resample,
                            mono=mono,
                            offset=start,
                            duration=STUDY_PARAMS["tr"],
                            dtype=args.dtype,
                            )
                        # only include complete audio chunks
                        if not audio_chunk.shape[-1] < n_frames :
                            if mono:
                                audio_segments.append(audio_chunk[:n_frames])
                            else:
                                audio_segments.append(audio_chunk[:,:n_frames])

                    flag = "a" if Path(out_file).exists() else "w"
                    with h5py.File(out_file, flag) as f:
                        group = f.create_group(episode)

                        for i in range(len(audio_segments)):
                            group.create_dataset(
                                f"{str(i).zfill(3)}",
                                data=audio_segments[i],
                                **comp_args,
                            )


if __name__ == "__main__":
    sys.exit(main())
