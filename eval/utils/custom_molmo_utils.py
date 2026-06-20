import base64
import copy
import logging
import math
import sys
import warnings
from functools import lru_cache
from io import BytesIO
from typing import Optional, Any
from concurrent.futures import ThreadPoolExecutor

import requests
import numpy as np
from PIL import ImageFile, ImageOps, Image


logger = logging.getLogger(__name__)

MAX_FRAMES: int = 384
FRAME_SAMPLE_MODE: str = "uniform_last_frame"
MAX_VIDEO_FPS: float = 8.0
SAMPLING_FPS: float = 2.0
MAX_FPS: int | float = 2
MAX_NUM_WORKERS_FETCH_VIDEO: int = 8


def setup_pil():
    Image.MAX_IMAGE_PIXELS = None
    ImageFile.LOAD_TRUNCATED_IMAGES = True


def fetch_image(ele: dict[str, Any]) -> Image.Image:
    setup_pil()

    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        # fix memory leak issue while using BytesIO
        with requests.get(image, stream=True) as response:
            response.raise_for_status()
            with BytesIO(response.content) as bio:
                image_obj = copy.deepcopy(Image.open(bio))
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            # fix memory leak issue while using BytesIO
            with BytesIO(data) as bio:
                image_obj = copy.deepcopy(Image.open(bio))
    else:
        image_obj = Image.open(image)
    
    if image_obj is None:
        raise ValueError(
            f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}"
        )
    
    with warnings.catch_warnings(record=True) as w:
        image_obj = image_obj.convert("RGB")
    try:
        image_obj = ImageOps.exif_transpose(image_obj)
    except Exception as e:
        pass

    return image_obj


def is_decord_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("decord") is not None


def is_torchcodec_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("torchcodec") is not None


@lru_cache(maxsize=1)
def get_default_video_reader_backend() -> str:
    if is_decord_available():
        video_reader_backend = "decord"
    elif is_torchcodec_available():
        video_reader_backend = "torchcodec"
    else:
        video_reader_backend = "pyav"
    print(f"molmo-utils using {video_reader_backend} by default to read video.", file=sys.stderr)
    return video_reader_backend


def get_candidate_target_fps(
    video_fps: float,
    sampling_fps: float,
    max_fps: float = MAX_VIDEO_FPS,
) -> list[float]:
    """
    Return the subset of `video_fps` factors that remain multiples of `sampling_fps`.

    Examples:
        >>> get_candidate_target_fps(video_fps=6, sampling_fps=2)
        [2, 6]
        >>> get_candidate_target_fps(video_fps=5, sampling_fps=1)
        [1, 5]
        >>> get_candidate_target_fps(video_fps=2, sampling_fps=2)
        [2]
        >>> get_candidate_target_fps(video_fps=5, sampling_fps=2)
        Traceback (most recent call last):
            ...
        ValueError: sampling_fps=2 must divide video_fps=5 to produce consistent frame steps.
    """
    video_fps = int(video_fps)
    sampling_fps = int(sampling_fps)
    max_fps = int(max_fps)

    if sampling_fps is None:
        raise ValueError("sampling_fps must be provided")
    if video_fps <= 0 or sampling_fps <= 0:
        raise ValueError(f"video_fps and sampling_fps must be positive (got {video_fps}, {sampling_fps})")
    if video_fps % sampling_fps != 0:
        raise ValueError(f"sampling_fps={sampling_fps} must divide video_fps={video_fps}.")

    candidates = []
    for candidate in range(sampling_fps, video_fps + 1, sampling_fps):
        if candidate > max_fps:
            break
        if video_fps % candidate == 0:
            candidates.append(float(candidate))
    return candidates


def sample_times(
    duration: float,
    num_frames: int,
    frame_sample_mode: str,
    candidate_target_fps: Optional[tuple[float, ...]] = None,
    max_fps: Optional[int | float] = None,
) -> np.ndarray:

    if frame_sample_mode == "uniform_last_frame":
        if max_fps is not None:
            max_duration = (num_frames-1) / max_fps  # -1 to include the last frame
            if max_duration < duration:
                times = np.linspace(0, duration, num=num_frames, endpoint=True, dtype=np.float64)
            else:
                times = np.arange(0.0, stop=duration, step=1/max_fps)
                times = np.concatenate([times, [duration]], axis=0)
                assert len(times) <= num_frames
        else:
            times = np.linspace(0, duration, num=num_frames, endpoint=True, dtype=np.float64)
        return times
    elif frame_sample_mode == "fps":
        # Try larger and larger FPSs until we hit one that can't span the video
        target_fps = candidate_target_fps[0]
        for candidate_fps in candidate_target_fps[1:]:
            if num_frames/candidate_fps < duration:
                break
            target_fps = candidate_fps
        times = np.arange(0, num_frames) / target_fps
        times = times[times < duration]
        return times
    else:
        raise NotImplementedError(frame_sample_mode)


def _validate_clip(clip: tuple[float, float], duration: float):
    if clip[0] >= clip[1]:
        raise ValueError(f"Clip {clip} has start>=end")
    if clip[0] >= duration:
        raise ValueError(f"Invalid clip, start={clip[0]} but video duration={duration}")


def load_video_decord(ele: dict[str, Any]) -> tuple[np.ndarray, dict]:
    """load video using decord.VideoReader

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - clip (tuple): the start and end time of clip.
            - num_frames (int): the max number of frames.
            - frame_sample_mode (str): the mode of frame sampling.
            - sampling_fps (Optional[float]): Rate to sample points at, in frames per second.
                Used for `frame_sample_mode` == "fps"
            - max_fps (Optional[float]): the maximum fps to sample. Default to 2.
    Returns:
        tuple[`np.array`, `VideoMetadata`]: A tuple containing:
            - Numpy array of frames in RGB (shape: [num_frames, height, width, 3]).
            - A dictionary containing the metadata of the video.
    """

    import decord
    video_path = ele["video"]
    clip = ele.get("clip", None)
    num_frames = ele.get("num_frames", MAX_FRAMES)
    frame_sample_mode = ele.get("frame_sample_mode", FRAME_SAMPLE_MODE)
    max_fps = ele.get("max_fps", MAX_FPS)
    vr = decord.VideoReader(video_path, num_threads=1, ctx=decord.cpu(0))
    video_fps = vr.get_avg_fps()
    candidate_target_fps: Optional[tuple[float, ...]] = None
    if frame_sample_mode == "fps":
        sampling_fps = ele.get("sampling_fps", SAMPLING_FPS)
        candidate_target_fps = get_candidate_target_fps(video_fps, sampling_fps)

    time_stamps = vr.get_frame_timestamp(list(range(len(vr))))
    duration = time_stamps[-1][1] - time_stamps[0][0]
    if clip:
        _validate_clip(clip, duration)
        clip_duration = min(clip[1], duration) - clip[0]
        target_timestamps = sample_times(
            clip_duration,
            num_frames,
            frame_sample_mode,
            candidate_target_fps,
            max_fps,
        )
        target_timestamps = np.array(target_timestamps)
        offset = clip[0]
        start_frame = math.floor(clip[0] * video_fps)
        end_frame = min(math.ceil(clip[1] * video_fps), len(vr))
        total_num_frames = end_frame - start_frame
        metadata = dict(
            total_num_frames=total_num_frames,
            fps=float(video_fps),
            duration=float(clip_duration),
            video_backend="decord",
        )
    else:
        target_timestamps = sample_times(
            duration,
            num_frames,
            frame_sample_mode,
            candidate_target_fps,
            max_fps,
        )
        target_timestamps = np.array(target_timestamps)
        offset = time_stamps[0, 0]
        metadata = dict(
            total_num_frames=int(len(vr)),
            fps=float(video_fps),
            duration=float(duration),
            video_backend="decord",
        )
    ix = np.searchsorted(time_stamps[:, 1], target_timestamps + offset, side='right')
    ix = np.minimum(ix, len(time_stamps) - 1)
    frames = vr.get_batch(ix).asnumpy()
    metadata.update(
        {
            "frames_indices": target_timestamps * video_fps,
            "height": frames.shape[1],
            "width": frames.shape[2],
        }
    )
    return frames, metadata


def load_video_torchcodec(ele: dict[str, Any]) -> tuple[np.ndarray, dict]:
    """load video using torchcodec.VideoDecoder

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - clip (tuple): the start and end time of clip.
            - num_frames (int): the max number of frames.
            - frame_sample_mode (str): the mode of frame sampling.
            - sampling_fps (Optional[float]): Rate to sample points at, in frames per second.
                Used for `frame_sample_mode` == "fps".
            - max_fps (Optional[float]): the maximum fps to sample. Default to 2.
    Returns:
        tuple[`np.array`, `VideoMetadata`]: A tuple containing:
            - Numpy array of frames in RGB (shape: [num_frames, height, width, 3]).
            - A dictionary containing the metadata of the video.
    """
    import torchcodec
    video_path = ele["video"]
    clip = ele.get("clip", None)
    num_frames = ele.get("num_frames", MAX_FRAMES)
    frame_sample_mode = ele.get("frame_sample_mode", FRAME_SAMPLE_MODE)
    max_fps = ele.get("max_fps", MAX_FPS)
    decoder = torchcodec.decoders.VideoDecoder(video_path, num_ffmpeg_threads=1, device="cpu")
    video_fps = decoder.metadata.average_fps
    total_frames = decoder.metadata.num_frames

    candidate_target_fps: Optional[tuple[float, ...]] = None
    if frame_sample_mode == "fps":
        sampling_fps = ele.get("sampling_fps", SAMPLING_FPS)
        candidate_target_fps = get_candidate_target_fps(video_fps, sampling_fps)

    # If the first frame starts at > 0, we effectively clip the video starting at that time
    # since (most) video players would also skip to that time
    time_offset = decoder.metadata.begin_stream_seconds_from_content
    # Note this duration does assume we started playing at `time_offset`
    duration = decoder.metadata.duration_seconds

    if clip:
        _validate_clip(clip, duration)
        clip_duration = min(clip[1], duration) - clip[0]
        target_timestamps = sample_times(
            clip_duration,
            num_frames,
            frame_sample_mode,
            candidate_target_fps,
            max_fps,
        )
        time_offset += clip[0]
        start_index = math.floor(clip[0] * video_fps)
        end_index = min(math.ceil(clip[1] * video_fps), total_frames)
        metadata = dict(
            total_num_frames=(end_index - start_index),
            fps=float(video_fps),
            duration=float(clip_duration),
            video_backend="torchcodec",
            height=decoder.metadata.height,
            width=decoder.metadata.width,
        )
    else:
        target_timestamps = sample_times(
            duration,
            num_frames,
            frame_sample_mode,
            candidate_target_fps,
            max_fps,
        )
        metadata = dict(
            total_num_frames=int(total_frames),
            fps=float(video_fps),
            duration=float(duration),
            video_backend="torchcodec",
            height=decoder.metadata.height,
            width=decoder.metadata.width,
        )

    # Floating point/rounding issues might cause `target_timestamps` to be very slightly
    # out-of-bounds, to handle this we sanity check then clip them
    assert all(x >= 0 for x in target_timestamps)
    assert all(x < duration+1e-6 for x in target_timestamps)
    # 1e-6 padding since torchcodec can throw out-of-bounds errors even if you ask for the
    # exact boundary value, we should still get the first/last frame anyway
    max_timestamp = decoder.metadata.end_stream_seconds_from_content - 1e-6
    min_timestamp = decoder.metadata.begin_stream_seconds_from_content + 1e-6
    # Note we avoid using numpy ops here to reduce floating precision issues
    timestamps = [x + time_offset for x in target_timestamps]
    timestamps = [max(min_timestamp, min(max_timestamp, x)) for x in timestamps]
    frames = decoder.get_frames_played_at(timestamps)
    target_timestamps = np.array(target_timestamps)
    metadata["frames_indices"] = target_timestamps * video_fps
    
    frames = frames.data.numpy().transpose(0, 2, 3, 1)  # Convert to THWC format

    return frames, metadata


def load_video_pyav_noseek(ele: dict[str, Any]) -> tuple[np.ndarray, dict]:
    """Load a video frames by decoding all frames with PyAV

    More robust than `load_video_decord` but can be much slower for long videos
    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - clip (tuple): the start and end time of clip.
            - num_frames (int): the max number of frames.
            - frame_sample_mode (str): the mode of frame sampling.
            - sampling_fps (Optional[float]): Rate to sample points at, in frames per second.
                Used for `frame_sample_mode` == "fps"
            - max_fps (Optional[float]): the maximum fps to sample. Default to 2.
    Returns:
        tuple[`np.array`, `VideoMetadata`]: A tuple containing:
            - Numpy array of frames in RGB (shape: [num_frames, height, width, 3]).
            - A dictionary containing the metadata of the video.
    """
    import av
    video_path = ele["video"]
    if "http://" in video_path or "https://" in video_path:
        raise ValueError("av does not support http/https video path")
    if "file://" in video_path:
        video_path = video_path[7:]
    clip = ele.get("clip", None)
    num_frames = ele.get("num_frames", MAX_FRAMES)
    frame_sample_mode = ele.get("frame_sample_mode", FRAME_SAMPLE_MODE)
    max_fps = ele.get("max_fps", MAX_FPS)

    # Behaves the same as the old version using `imageio.v3` but avoid extra the dependency
    with av.open(video_path) as container:
        video_stream = container.streams.video[0]
        fps = video_stream.average_rate or video_stream.guessed_rate
        candidate_target_fps: Optional[tuple[float, ...]] = None
        if frame_sample_mode == "fps":
            sampling_fps = ele.get("sampling_fps", SAMPLING_FPS)
            candidate_target_fps = get_candidate_target_fps(fps, sampling_fps)
        it = container.decode(video=0)
        frames = list(it)

        stream = container.streams.video[0]
        start = frames[0].pts * stream.time_base
        container_end = stream.duration
        if container_end is not None:
            container_end *= stream.time_base
        if container_end is None or container_end < frames[-1].pts:
            # Some problem with stream duration, so use the frame PTS directly
            # and guess the duration of the last frame
            end = frames[-1].pts * stream.time_base + 1/fps
        else:
            end = container_end
        duration = float(end - start)
        if clip is not None:
            _validate_clip(clip, duration)
            clip_duration = min(clip[1], duration)-clip[0]
            timestamps = sample_times(
                clip_duration,
                num_frames,
                frame_sample_mode,
                candidate_target_fps,
                max_fps,
            )
            offset = clip[0]
            start_index = math.floor(clip[0] * fps)
            end_index = min(math.ceil(clip[1] * fps), len(frames))
            metadata = dict(
                total_num_frames=(end_index - start_index),
                fps=float(fps),
                duration=float(clip_duration),
                video_backend="pyav",
                height=video_stream.height,
                width=video_stream.width,
            )
        else:
            timestamps = sample_times(
                duration,
                num_frames,
                frame_sample_mode,
                candidate_target_fps,
                max_fps,
            )
            offset = float(start)
            metadata = dict(
                total_num_frames=int(len(frames)),
                fps=float(fps),
                duration=float(duration),
                video_backend="pyav",
                height=video_stream.height,
                width=video_stream.width,
            )
        timestamps = np.array(timestamps)
        end_time_stamps = np.array([float(frame.pts * stream.time_base) for frame in frames[1:]] + [duration])
        indices = np.searchsorted(end_time_stamps, timestamps + offset, side='right')
        indices = np.minimum(indices, len(end_time_stamps) - 1)
        metadata["frames_indices"] = timestamps * fps

        frames = [frames[i].to_ndarray(format="rgb24", channel_last=True) for i in indices]
        
        return frames, metadata


VIDEO_READER_BACKENDS = {
    "decord": load_video_decord,
    "torchcodec": load_video_torchcodec,
    "pyav": load_video_pyav_noseek,
}


def fetch_video(ele: dict[str, Any]) -> tuple[np.ndarray, dict]:
    if isinstance(ele["video"], str):
        video_reader_backend = ele.get("backend", get_default_video_reader_backend())
        if video_reader_backend == "torchcodec":
            video, video_metadata = load_video_torchcodec(ele)
        else:
            try:
                video, video_metadata = VIDEO_READER_BACKENDS[video_reader_backend](ele)
            except Exception as e:
                warnings.warn(
                    f"Failed to decode video with {video_reader_backend}: {e}"
                    "Falling back to `PyAV`."
                )
                video, video_metadata = VIDEO_READER_BACKENDS["pyav"](ele)
    else:
        # The input is a list of frames
        assert isinstance(ele["video"], (list, tuple))
        max_workers = min(MAX_NUM_WORKERS_FETCH_VIDEO, len(ele["video"]))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(fetch_image, {"image": video_element})
                for video_element in ele["video"]
            ]
            image_list = [future.result() for future in futures]
        
        video = np.stack([np.array(image) for image in image_list])
        timestamps = ele.get("timestamps", None)
        if timestamps is None:
            raise ValueError("timestamps is required when video is a list of images")
        timestamps = np.array(timestamps)
        fps = 1 / (timestamps[1:] - timestamps[:-1]).mean()
        video_metadata = dict(
            total_num_frames=len(video),
            fps=fps,
            frames_indices=timestamps * fps,
            height=video.shape[1],
            width=video.shape[2],
        )
    return video, video_metadata


def extract_vision_info(
    conversations: list[dict[str, Any]] | list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], (list, tuple)):
                for ele in message["content"]:
                    if (
                        "image" in ele
                        or "image_url" in ele
                        or "video" in ele
                        or ele.get("type", "text") in ("image", "image_url", "video")
                    ):
                        vision_infos.append(ele)
    return vision_infos


def process_vision_info(
    conversations: list[dict[str, Any]] | list[list[dict[str, Any]]],
) -> tuple[Optional[list[Image.Image]], Optional[list[tuple[np.ndarray, dict]]], dict[str, Any]]:
    vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info))
        elif "video" in vision_info:
            video_inputs.append(fetch_video(vision_info))
        else:
            raise ValueError("image, image_url or video should be in content")
    
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    
    video_kwargs = {'do_sample_frames': False}
    return image_inputs, video_inputs, video_kwargs