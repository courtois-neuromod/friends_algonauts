import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


STUDY_PARAMS = {
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
        "--seasons",
        default=None,
        nargs="+",
        help="List of seasons of .mkv files to process. If none is specified, "
        "all seasons will be processed.",
    )

    return parser.parse_args()


def main() -> None:
    """.

    This script loads .tsv files of script words force-aligned to the movie audio,
    and converts their onset and duration from seconds to TRs.
    """
    args = get_arguments()

    print(vars(args))

    seasons = (
        args.seasons
        if args.seasons is not None
        else [x.split("/")[-1] for x in sorted(glob.glob(f"{args.idir}/s*"))]
    )
    print("Seasons : ", seasons)

    for season in seasons:
        episode_list = [
            x.split("/")[-1].split(".")[0][-7:]
            for x in sorted(glob.glob(f"{args.idir}/{season}/aligned_s*.tsv"))
        ]
        print(episode_list)

        Path(f"{args.odir}/{season}").mkdir(parents=True, exist_ok=True)

        for episode in tqdm(episode_list, desc="processing .tsv files"):
            in_file = f"{args.idir}/{season}/aligned_{episode}.tsv"
            out_file = f"{args.odir}/{season}/friends_{episode}_aligned-script.tsv"

            df = pd.read_csv(in_file, sep='\t')
            df['onset_TRs'] = df.apply(lambda row: row['onset']/STUDY_PARAMS["tr"], axis=1)
            df['duration_TRs'] = df.apply(lambda row: row['duration']/STUDY_PARAMS["tr"], axis=1)

            df.to_csv(out_file, sep='\t', header=True, index=False)


if __name__ == "__main__":
    sys.exit(main())
