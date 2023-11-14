import argparse
import gc
import glob
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet.gluon.data.vision import transforms
from gluoncv.data import transforms
from gluoncv.model_zoo import get_model
from gluoncv.model_zoo.action_recognition.i3d_resnet import I3D_ResNetV1
from gluoncv.data import VideoClsCustom
from gluoncv.utils.filesystem import try_import_decord
import h5py
import IPython.display as ipd
import librosa
import librosa.display
#from moviepy.editor import *
from moviepy.editor import VideoFileClip
#import torch
from tqdm import tqdm

# https://github.com/deepinsight/insightface/issues/694
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

STUDY_PARAMS = {
    "fps": 29.97,
    "tr": 1.49,
    "sr": 22050,  # orig is 44100 for Friends
    "resized_height": 224,  # default is 256,
    "crop_size": 224,  # defaultis 224
    "num_segments": 1,
    "num_crop": 1,
    "slowfast": True,
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
        "--time_downsample",
        type=int,
        default=1,
        choices=range(1, 45),
        metavar="[1-44]",
        help="Temoral downsampling factor (1, 45). "
        "1 = no downsampling. 3 = keeps 1 out of 3 frames.",
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
        f"visual_audio{compress_details}.h5"
    )

    Path(f"{args.odir}/temp").mkdir(exist_ok=True, parents=True)

    return comp_args, out_file


def extract_audio_features(
    odir: str,
    season: str,
    sr: int,
) -> np.array:
    """.

    Calculates the Mel-frequency cepstral coefficients (MFCCs)
    from the audio signal with librosa.
    https://librosa.org/doc/main/generated/librosa.feature.mfcc.html
    """
    y, sr = librosa.load(
        f"{odir}/temp/audio_{season}.mp3",
        sr=sr,
        mono=True,
    )

    return np.mean(librosa.feature.mfcc(y=y, sr=sr), axis = 1)


def get_ctx():
    gpu_id = 0
    if gpu_id == -1:
        return mx.cpu()
    else:
        return mx.gpu(gpu_id)


def build_I3D_ResNet(
    context,
) -> I3D_ResNetV1:
    """.

    Instantiate pre-trained inflated 3D model (I3D) with ResNet50
    backbone trained on Kinetics400 dataset with gluoncv.
    https://cv.gluon.ai/build/examples_torch_action_recognition/demo_i3d_kinetics400.html
    """
    # set garbage collection threshold
    gc.set_threshold(100, 5, 5)

    net = get_model(
            name = 'i3d_resnet50_v1_kinetics400',
            nclass = 400,
            pretrained = True,
            feat_ext = True,
            num_segments = STUDY_PARAMS["num_segments"],
            num_crop = STUDY_PARAMS["num_crop"],
            ctx = context,
        )
    net.cast('float32')
    net.collect_params().reset_ctx(context)
    #net.eval()

    return net


def get_resize_dims(
    new_step: int,
) -> tuple:
    """
    WIP. Set frame resizing parameters.
    The default (movie10) pipeline resizes frame height to 256 and width to 340 pixels,
    then crops a 224 x 224 square in the center.

    Here, I resize the height to 224 pixels and then crop along the width
    to remove less info on each side and none along the height.

    For the CNeuroMod movie10 dataset, the aspect ratio is proportional to
    the frame size in pixels.
    Note that this isn't the case for Friends frames (w = 720, h = 480 pixels;
    on screen aspect rario = 4 / 3.0). Here, I resize the image width
    proportionally to the aspect ratio to respect image proportions as seen on
    the screen, rather than the original frame proportions in pixels.
    """
    crop_size = STUDY_PARAMS["crop_size"]  # default is 224
    rs_height = STUDY_PARAMS['resized_height']  # default height = 256

    # Resize proportionally to frame size in pixels
    #rs_width = int((rs_height * 720) / 480)

    # Resize proportionally to Friends on-screen aspect ratio: 4/3.0
    rs_width = int((rs_height * 4) / 3)

    """
    # Validate/adjust new lenght... (frames per chunk if slowfast is False)
    # https://github.com/dmlc/gluon-cv/blob/567775619f3b97d47e7c360748912a4fd883ff52/gluoncv/data/video_custom/classification.py#L44
    movie10 pipeline new_length = 32; ~35 frames per TR; Friends is ~44 frames / TRs
    """
    n_frames = int(STUDY_PARAMS["tr"] * STUDY_PARAMS["fps"])
    new_length = int(n_frames/new_step)

    return crop_size, rs_height, rs_width, new_length


def extract_visual_features(
    odir: str,
    season: str,
    #net: I3D_ResNetV1,
    new_step: int,
) -> np.array:
    """.

    Derive visual features from consecutive movie frames fed to
    a pre-trained inflated 3D convnet implemented in gluoncv.
    """
    video_path = f"{odir}/temp/clip_{season}.mp4"

    # Get resize & downsampling parameters
    num_segments = STUDY_PARAMS["num_segments"]
    num_crop = STUDY_PARAMS["num_crop"]
    slowfast = STUDY_PARAMS["slowfast"]

    crop_size, rs_height, rs_width, new_length = get_resize_dims(new_step)

    # Resize and downsample frames
    video_utils = VideoClsCustom(
                                root='',
                                setting='',
                                num_segments = num_segments,
                                num_crop = num_crop,
                                new_length = new_length,
                                new_step = new_step,
                                new_width = rs_width,
                                new_height = rs_height,
                                video_loader = True,
                                use_decord = True,
                                slowfast = slowfast,
                                slow_temporal_stride = 16, # default = 16
                                fast_temporal_stride = 2, # default = 2
                                data_aug = "v1",
                                lazy_init = True)

    decord = try_import_decord()
    decord_vr = decord.VideoReader(video_path, width=rs_width, height=rs_height)
    duration = len(decord_vr)

    segment_indices, skip_offsets = video_utils._sample_test_indices(duration)

    if slowfast:
        clip_input = video_utils._video_TSN_decord_slowfast_loader(
            video_path,
            decord_vr,
            duration,
            segment_indices,
            skip_offsets,
        )
    else:
        clip_input = video_utils._video_TSN_decord_batch_loader(
            video_path,
            decord_vr,
            duration,
            segment_indices,
            skip_offsets,
        )

    # Normalize with ImageNet norms (per color channel) and center crop
    transform_test = transforms.video.VideoGroupValTransform(
        size = crop_size,
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225],
    )
    clip_input = transform_test(clip_input)

    # Re-orient frame chunk for convnet
    if slowfast:
        sparse_samples = len(clip_input) // (num_segments * num_crop)
        clip_input = np.stack(clip_input, axis=0)
        clip_input = clip_input.reshape((-1,) + (sparse_samples, 3, crop_size, crop_size))
        clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))
    else:
        clip_input = np.stack(clip_input, axis=0)
        clip_input = clip_input.reshape((-1,) + (new_length, 3, crop_size, crop_size))
        clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))

    if new_length == 1:
        clip_input = np.squeeze(clip_input, axis=2)

    """
    Obtain features from pre-trained convnet for frame chunk
    https://github.com/dmlc/gluon-cv/blob/567775619f3b97d47e7c360748912a4fd883ff52/scripts/action-recognition/feat_extract_pytorch.py#L45C30-L45C30

    TODO: set model to .eval() to freeze pre-trained params (bug?)
    so that one model can process the whole dataset...
    """
    context = get_ctx()
    video_input = nd.array(clip_input).as_in_context(context)
    net = build_I3D_ResNet(context)

    #with torch.no_grad():
    video_feat = net(video_input.astype('float32', copy=False))

    return np.squeeze(video_feat.asnumpy())


def extract_features(
    season: str,
    mkv_path: str,
    #net: I3D_ResNetV1,
    args: argparse.Namespace,
) -> tuple:
    """.

    Extract audio and visual features from an episode, in chunks.
    """
    if Path(mkv_path).exists():

        clip = VideoFileClip(mkv_path)
        start_times = [x for x in np.arange(0, clip.duration, STUDY_PARAMS["tr"])][:-1]

        visual_features = []
        audio_features = []

        for start in start_times:
            clip_chunk = clip.subclip(start, start + STUDY_PARAMS["tr"])
            clip_chunk.write_videofile(
                f"{args.odir}/temp/clip_{season}.mp4",
                verbose=False,
            )
            clip_chunk.audio.write_audiofile(
                f"{args.odir}/temp/audio_{season}.mp3",
                verbose=False,
            )

            chunk_visual_feat = extract_visual_features(
                args.odir,
                season,
                #net,
                args.time_downsample,
            )
            visual_features.append(chunk_visual_feat)

            audio_features.append(
                extract_audio_features(
                    args.odir,
                    season,
                    STUDY_PARAMS["sr"],
                ),
            )

        vis_feat = np.array(
            visual_features,
            dtype='float32',
        )
        aud_feat = np.array(
            audio_features,
            dtype='float32',
        )

        return vis_feat, aud_feat


def save_features(
    episode: str,
    features: tuple,
    outfile_name: str,
    comp_args: dict,
) -> None:
    """.

    Save episode's audio and visual features into .h5 file.
    """
    flag = "a" if Path(outfile_name).exists() else "w"

    with h5py.File(outfile_name, flag) as f:
        group = f.create_group(episode)

        group.create_dataset(
            "visual",
            data=features[0],
            **comp_args,
            )

        group.create_dataset(
            "audio",
            data=features[1],
            **comp_args,
        )


def process_episodes(
    season: str,
    #net: I3D_ResNetV1,
    args: argparse.Namespace,
) -> None:
    """.

    Extract audio and visual features from a season's episodes.
    """
    # set compression params and output file
    comp_args, outfile_name = set_output(season, args)

    episode_list = list_episodes(args.idir, season, outfile_name, args.verbose)

    for episode in tqdm(episode_list, desc="processing .mkv files"):
        mkv_path = f"{args.idir}/{season}/friends_{episode}.mkv"

        features = extract_features(
            season,
            mkv_path,
            #net,
            args,
        )
        save_features(
            episode,
            features,
            outfile_name,
            comp_args,
        )


def main() -> None:
    """.

    This script extracts features from movies to perform brain encoding
    on the Courtois Neuromod friends dataset. Movies are half-episodes
    of the sitcom Friends that were shown to participants as they
    underwent fMRI.

    The script derives visual and audio features from .mkv files
    broken into chunks that correspond to 1.49s of movie watching.

    This duration corresponds to the temporal frequency at which
    fMRI brain volumes were acquired for this dataset (TR = 1.49s),
    to facilitate the pairing between video features and brain data.

    TR-aligned features are saved into .h5 files and can be loaded
    as data matrices to predict runs of fMRI activity.

    Visual features are produced by feeding sequences of movie frames
    to an inflated 3D ResNet pre-trained to recognize actions on Kinetics400.

    Audio features are the Mel-frequency cepstral coefficients (MFCCs)
    calculated from the audio signal with librosa.

    Credit: Adapted from
    https://github.com/jashna14/DL4Brain/blob/master/src/Feature_extraction.py
    """

    args = get_arguments()
    print(vars(args))

    # Instantiate neural net to derive visual features
    #net = build_I3D_ResNet(get_ctx())

    # Get list of seasons to process
    seasons = list_seasons(args.idir, args.seasons)
    print("Seasons : ", seasons)

    for season in seasons:
        process_episodes(
            season,
            #net,
            args,
        )


if __name__ == "__main__":
    sys.exit(main())
