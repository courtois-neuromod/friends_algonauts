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
from gluoncv.data.transforms import video
from gluoncv.model_zoo import get_model
from gluoncv.data import VideoClsCustom
from gluoncv.utils.filesystem import try_import_decord
import h5py
import IPython.display as ipd
import librosa
import librosa.display
#from moviepy.editor import *
from moviepy.editor import VideoFileClip
from tqdm import tqdm


STUDY_PARAMS = {
    "fps": 29.97,
    "tr": 1.49,
    "sr": 22050, # orig is 44100 for Friends
    "height": 224, #256,
    # "crop_size": 224, orig pipeline resizes the height to 256 and width proportionally, then crops (square: 224x224)
    # I resized to 224 and then crop along width to remove less info on the sides and none along the height
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


def extract_audio_features(odir, sr):
    y,sr = librosa.load(
        f"{odir}/temp/audio.mp3",
        sr=sr,
        mono=True,
    )
    #y = librosa.to_mono(y)
    # https://librosa.org/doc/main/generated/librosa.feature.mfcc.html
    mfcc = librosa.feature.mfcc(y=y,sr=sr)
    mfcc = np.mean(mfcc,axis = 1)
    return mfcc


def extract_visual_features(odir, new_height, new_width, new_step):

    gc.set_threshold(100, 5, 5)
    gpu_id = 0
    input_size = 224
    use_pretrained = True
    hashtag = ''
    num_classes = 400
    # https://cv.gluon.ai/build/examples_torch_action_recognition/demo_i3d_kinetics400.html
    model_name  = 'i3d_resnet50_v1_kinetics400'
    num_segments = 1
    resume_params = ''
    video_path = f"{odir}/temp/clip.mp4"
    #new_height = 256
    #new_width = 340 # for Friends, should be 384 to be proportional

    # TODO: validate/adjust new lenght...
    # https://github.com/dmlc/gluon-cv/blob/567775619f3b97d47e7c360748912a4fd883ff52/gluoncv/data/video_custom/classification.py#L44
    new_length = int(44/new_step) # 32 # for movie 10, 35ish frames per TR... Friends is 44ish
    video_loader = True
    use_decord = True
    slowfast = True
    slow_temporal_stride = 16 # TODO: 11? 44/4 = 11...
    fast_temporal_stride = 2
    data_aug = 'v1'

  # set env
    if gpu_id == -1:
        context = mx.cpu()
    else:
        context = mx.gpu(gpu_id)

  # get data preprocess
  # notmalized to Imagenet norms, greyscale (1 color channel), since pre-trained
    image_norm_mean = [0.485, 0.456, 0.406]
    image_norm_std = [0.229, 0.224, 0.225]
  # https://github.com/dmlc/gluon-cv/blob/567775619f3b97d47e7c360748912a4fd883ff52/gluoncv/data/transforms/video.py#L82
    transform_test = video.VideoGroupValTransform(size=input_size, mean=image_norm_mean, std=image_norm_std)
    num_crop = 1

  # get model
    if use_pretrained and len(hashtag) > 0:
        use_pretrained = hashtag
    net = get_model(
            name=model_name,
            nclass=num_classes,
            pretrained=use_pretrained,
            feat_ext=True,
            num_segments=num_segments,
            num_crop=num_crop,
        )
    net.cast('float32')
    net.collect_params().reset_ctx(context)
    if resume_params != '' and not use_pretrained:
        net.load_parameters(resume_params, ctx=context)

    # TODO: debug
    # https://github.com/dmlc/gluon-cv/blob/master/gluoncv/data/video_custom/classification.py#L12
    # https://github.com/dmlc/gluon-cv/blob/567775619f3b97d47e7c360748912a4fd883ff52/gluoncv/data/video_custom/classification.py#L12
    # https://github.com/dmlc/gluon-cv/blob/567775619f3b97d47e7c360748912a4fd883ff52/gluoncv/utils/filesystem.py#L24
    video_utils = VideoClsCustom(
                                root='',
                                setting='',
                                num_segments=num_segments,
                                num_crop=num_crop,
                                new_length=new_length,
                                new_step=new_step,
                                new_width=new_width,
                                new_height=new_height,
                                video_loader=video_loader,
                                use_decord=use_decord,
                                slowfast=slowfast,
                                slow_temporal_stride=slow_temporal_stride,
                                fast_temporal_stride=fast_temporal_stride,
                                data_aug=data_aug,
                                lazy_init=True)

    decord = try_import_decord()
    decord_vr = decord.VideoReader(video_path, width=new_width, height=new_height)
    duration = len(decord_vr)

    #skip_length = new_length * new_step #yTorch
    segment_indices, skip_offsets = video_utils._sample_test_indices(duration)

    if video_loader:
        if slowfast:
            clip_input = video_utils._video_TSN_decord_slowfast_loader(video_path, decord_vr, duration, segment_indices, skip_offsets)
        else:
            clip_input = video_utils._video_TSN_decord_batch_loader(video_path, decord_vr, duration, segment_indices, skip_offsets)
    else:
        raise RuntimeError('We only support video-based inference.')

    clip_input = transform_test(clip_input)

    if slowfast:
        sparse_samples = len(clip_input) // (num_segments * num_crop)
        clip_input = np.stack(clip_input, axis=0)
        clip_input = clip_input.reshape((-1,) + (sparse_sampels, 3, input_size, input_size))
        clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))
    else:
        clip_input = np.stack(clip_input, axis=0)
        clip_input = clip_input.reshape((-1,) + (new_length, 3, input_size, input_size))
        clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))

    if new_length == 1:
        clip_input = np.squeeze(clip_input, axis=2)    # this is for 2D input case

    video_data = nd.array(clip_input)
    video_input = video_data.as_in_context(context)
    video_feat = net(video_input.astype('float32', copy=False))

    return np.squeeze(video_feat.asnumpy())


def main() -> None:
    """.

    This script derives visual and frame features from .mkv files
    into chunks that correspond to 1.49s of movie watching.
    This duration matches the temporal frequency at which
    fMRI brain volumes were acquired for the current dataset
    (TR = 1.49s), to facilitate the pairing between movie frames
    and brain data.

    Chunks are saved into .h5 files that can be loaded into data matrices
    to predict brain activity.

    Use pre-trained ResNet to output visual features, and librosa
    to process audio signal
    Credit: based on
    https://github.com/jashna14/DL4Brain/blob/master/src/Feature_extraction.py#L129
    """
    # https://github.com/deepinsight/insightface/issues/694
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

    args = get_arguments()
    #dtype = "float32" if args.normalize else "uint8"

    print(vars(args))

    rs_height = STUDY_PARAMS['height']
    #rs_width = int((rs_height * 720) / 480)  # from frame's original size in pixels
    rs_width = int((rs_height * 4) / 3)  # from Friends's aspect ratio: 4/3.0
    # note that, for movie10, the aspect ratio is proportional to frame size in pixels... (not for Friends)

    n_frames_float = STUDY_PARAMS["tr"] * STUDY_PARAMS["fps"]
    n_frames = int(n_frames_float)

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
            f"{args.odir}/friends_{season}_features_visual_audio_{compress_details}.h5"
        )
        Path(f"{args.odir}/temp").mkdir(exist_ok=True, parents=True)

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

                    clip = VideoFileClip(mkv_path)
                    start_times = [x for x in np.arange(0, clip.duration, STUDY_PARAMS["tr"])][:-1]

                    visual_features = []
                    audio_features = []

                    for start in start_times:
                        clip_chunk = clip.subclip(start, start + STUDY_PARAMS["tr"])
                        clip_chunk.write_videofile(f"{args.odir}/temp/clip.mp4",verbose=False)
                        clip_chunk.audio.write_audiofile(f"{args.odir}/temp/audio.mp3",verbose=False)

                        chunk_visual_feat = extract_visual_features(
                            args.odir,
                            rs_height,
                            rs_width,
                            args.time_downsample,
                        )
                        visual_features.append(chunk_visual_feat)
                        audio_features.append(
                            extract_audio_features(
                                args.odir,
                                STUDY_PARAMS["sr"],
                            ),
                        )

                    flag = "a" if Path(out_file).exists() else "w"
                    with h5py.File(out_file, flag) as f:
                        group = f.create_group(episode)

                        group.create_dataset(
                            "visual",
                            data=np.array(
                                    visual_features,
                                    dtype='float32',
                                ),
                            **comp_args,
                        )

                        group.create_dataset(
                            "audio",
                            data=np.array(
                                    audio_features,
                                    dtype='float32',
                                ),
                            **comp_args,
                        )
                        

if __name__ == "__main__":
    sys.exit(main())
