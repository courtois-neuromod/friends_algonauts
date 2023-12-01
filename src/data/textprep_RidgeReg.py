import argparse
import glob
import string
import sys
from pathlib import Path

from transformers import BertTokenizer, BertModel
from transformers import pipeline

import numpy as np
import pandas as pd
import h5py
import torch
from tqdm import tqdm

STUDY_PARAMS = {
    "tr": 1.49,
    "max_tokens": 512,
}

def get_arguments() -> argparse.Namespace:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Derives visual and audio features from movie chunks (.mkv)"
            " and export them into .h5 files (one per season)."
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
        " .h5 files will be saved.",
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
        help="Compression to apply to features. Default is none.",
    )
    parser.add_argument(
        "--compression_opts",
        type=int,
        default=4,
        choices=range(0, 10),
        help="Feature compression level in .h5 file. Value = [0-9]. "
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
    seasons: list=None,
) -> list:
    """.

    List of seasons to process.
    """
    season_list = (
        seasons if seasons is not None
        else [x.split("/")[-1] for x in sorted(glob.glob(f"{idir}/s*"))]
    )

    return season_list


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
        x.split("/")[-1].split(".")[0][8:15]
        for x in sorted(glob.glob(f"{idir}/{season}/friends_s*.tsv"))
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


def set_output(
    season: str,
    args: argparse.Namespace,
) -> tuple:
    """.

    Set compression params and output file name.
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
        f"{args.odir}/friends_{season}_features_"
        f"text{compress_details}.h5"
    )

    #Path(f"{args.odir}/temp").mkdir(exist_ok=True, parents=True)

    return comp_args, out_file


def extract_features(
    tsv_path: str,
    model: BertModel,
    tokenizer: BertTokenizer,
    args: argparse.Namespace,
) -> np.array:
    """.
    LINKS OPEN-AI (gpt-4)
    https://platform.openai.com/docs/quickstart?context=python
    https://platform.openai.com/docs/introduction
    https://platform.openai.com/docs/guides/embeddings/what-are-embeddings
    https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_long_inputs.ipynb
    https://github.com/openai/tiktoken
    https://medium.com/vectrix-ai/gpt-4-chatbot-guide-mastering-embeddings-and-personalized-knowledge-bases-f58290e81cf4



    LINKS BERT:
    Extract text features from an episode's transcript, in chunks of 1 TR.
    https://huggingface.co/bert-base-uncased
    https://github.com/jashna14/DL4Brain/blob/master/src/Feature_extraction.py
    https://colab.research.google.com/drive/1w1a08G9zNVNANtiEZNCdMAi5M3eO_yew#scrollTo=kDs-MYtYH8sL
    https://huggingface.co/docs/transformers/preprocessing#build-tensors

    source code: https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/tokenization_bert.py
    """
    # https://discuss.huggingface.co/t/extracting-token-embeddings-from-pretrained-language-models/6834/5
    #pipe = pipeline('feature-extraction', model=model, tokenizer=tokenizer)

    max_tk = 8

    df = pd.read_csv(tsv_path, sep = '\t')
    df.insert(loc=0, column="is_na", value=df["text_per_tr"].isna())

    #raw_text = ""
    tokens = []
    np_token = []
    token_features = []
    pooled_features = []

    for i in range(df.shape[0]):
        #print(i)
        num_tokens = 0
        if not df.iloc[i]["is_na"]:
            tr_text = df.iloc[i]["text_per_tr"]
            #raw_text += tr_text
            #print(tr_text)

            # tokenize raw punctuated text
            tokens.extend(tokenizer.tokenize(tr_text))

            tr_np_tokens = tokenizer.tokenize(
                tr_text.translate(str.maketrans('', '', string.punctuation)),
            )
            num_tokens = len(tr_np_tokens)
            np_token.extend(tr_np_tokens)

        if len(tokens) > 0:
            # for each TR, extract features from window <= 512 of the latest tokens
            input_ids = [101] + tokenizer.convert_tokens_to_ids(
                tokens[-(STUDY_PARAMS["max_tokens"]-2):]
            ) + [102]
            tensor_tokens = torch.tensor(input_ids).unsqueeze(0)

            with torch.no_grad():
                outputs = model(tensor_tokens)

            pooled_features.append(
                np.array(
                    outputs["pooler_output"][0].detach().numpy(),
                    dtype='float32',
                )
            )

            last_feat = np.repeat(np.nan, 768*max_tk).reshape((max_tk, 768))
            if num_tokens > 0:

                tk_idx = min(max_tk, num_tokens)
                # truncate raw text to last 510 tokens (BERT maximum)
                #np_tr_text = tokenizer.convert_tokens_to_string(np_token[-(STUDY_PARAMS["max_tokens"]-2):])
                #data = pipe(np_tr_text)
                #last_embeddings = np.array(data[0][-(tk_idx+1):-1], dtype='float32')
                input_ids_np = [101] + tokenizer.convert_tokens_to_ids(
                    np_token[-(STUDY_PARAMS["max_tokens"]-2):]
                ) + [102]
                np_tensor_tokens = torch.tensor(input_ids_np).unsqueeze(0)

                with torch.no_grad():
                    np_outputs = np.array(
                        model(
                            np_tensor_tokens
                        )['last_hidden_state'][0][1:-1].detach().numpy(),
                        dtype='float32',
                    )

                last_feat[-tk_idx:, :] = np_outputs[-tk_idx:]

            token_features.append(last_feat)

        else:
            token_features.append(
                np.repeat(np.nan, 768*max_tk).reshape((max_tk, 768))
            )
            pooled_features.append(
                np.repeat(np.nan, 768)
            )

    """
    # old pipeline for aligned-scripts.tsv files, one row per timestamped word...

    tokens = []
    features = []

    start_times = [x for x in np.arange(0, df['onset'].tolist()[-1] + 1.0, STUDY_PARAMS["tr"])]

    for start in start_times:
        words_tr = df[np.logical_and(df['onset'] > start, df['onset'] <= start+STUDY_PARAMS["tr"])]["word"].tolist()
        if len(words_tr) > 0:
            for word in words_tr:
                tokens.extend(tokenizer.tokenize(word))

        if len(tokens) > 0:
            input_ids = [101] + tokenizer.convert_tokens_to_ids(tokens[-STUDY_PARAMS["max_tokens"]:]) + [102]
            tr_tokens = torch.tensor(input_ids).unsqueeze(0)

            with torch.no_grad():
                outputs = model(tr_tokens)
            features.append(
                np.array(outputs["pooler_output"][0].detach().numpy(), dtype='float32')
            )

        else:
            features.append(
                np.repeat(np.nan, 768)
            )

    return np.concatenate(features, axis=0)
    """
    return np.array(pooled_features, dtype='float32'), np.array(token_features, dtype='float32')


def save_features(
    episode: str,
    pool_features: np.array,
    tk_features: np.array,
    outfile_name: str,
    comp_args: dict,
) -> None:
    """.

    Save episode's text features into .h5 file.
    """
    flag = "a" if Path(outfile_name).exists() else "w"

    with h5py.File(outfile_name, flag) as f:
        group = f.create_group(episode)

        group.create_dataset(
            "text_pooled",
            data=pool_features,
            **comp_args,
            )
        group.create_dataset(
            "text_token",
            data=tk_features,
            **comp_args,
            )


def process_episodes(
    season: str,
    args: argparse.Namespace,
) -> None:
    """.

    Extract text features from a season's episode transcripts.
    https://huggingface.co/bert-base-uncased
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()

    # set compression params and output file
    comp_args, outfile_name = set_output(season, args)

    episode_list = list_episodes(args.idir, season, outfile_name, args.verbose)

    for episode in tqdm(episode_list, desc="processing .tsv files"):
        tsv_path = f"{args.idir}/{season}/friends_{episode}_aligned-text.tsv"

        if Path(tsv_path).exists():
            pool_features, tk_features = extract_features(
                tsv_path,
                model,
                tokenizer,
                args,
            )
            save_features(
                episode,
                pool_features,
                tk_features,
                outfile_name,
                comp_args,
            )


def main() -> None:
    """.

    This script extracts features from movie transcripts to perform brain
    encoding on the Courtois Neuromod friends dataset. Transcripts are
    from half-episodes of the sitcom Friends that were shown to participants
    as they underwent fMRI.

    The script derives text features from .tsv files of time-stamped
    transcript words aligned to the movie's audio. Chunks of words
    that correspond to 1.49s of movie watching are used to produce
    embeddings with a pre-trained BERT model.

    The 1.49s chunk duration corresponds to the temporal frequency at which
    fMRI brain volumes were acquired for this dataset (TR = 1.49s),
    to facilitate the pairing between video features and brain data.

    TR-aligned features are saved into .h5 files and can be loaded
    as data matrices to predict runs of fMRI activity.

    Credit: Adapted from
    https://github.com/jashna14/DL4Brain/blob/master/src/Feature_extraction.py
    """

    args = get_arguments()
    print(vars(args))

    # Get list of seasons to process
    seasons = list_seasons(args.idir, args.seasons)
    print("Seasons : ", seasons)

    for season in seasons:
        process_episodes(
            season,
            args,
        )


if __name__ == "__main__":
    sys.exit(main())
