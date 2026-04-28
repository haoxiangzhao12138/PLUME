import os

from datasets import load_dataset
from src.data.dataset_hf_path import EVAL_DATASET_HF_PATH
from src.data.utils.dataset_utils import load_hf_dataset, sample_dataset

from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook, RESOLUTION_MAPPING
from src.data.eval_dataset.base_eval_dataset import ImageVideoInstance
from src.data.utils.vision_utils import sample_frames, load_frames, VID_EXTENSIONS, save_frames
from src.model.processor import process_input_text

TASK_INST_QRY_TEXT = "Find the clip that corresponds to the given text:"
TASK_INST_QRY_IMG = "Select the video clip that aligns with the given text and image:"
TASK_INST_QRY_VIDEO = "Find the clip that corresponds to the given sentence and video segment:"
TASK_INST_TGT = "Understand the content of the provided video clip."


def _resolve_momentseeker_frame_path(frame_root, rel_path):
    """Resolve frame path across legacy and current MMEB MomentSeeker layouts."""
    if not rel_path:
        return None

    candidates = [os.path.join(frame_root, rel_path)]
    if rel_path.startswith("videos/"):
        candidates.append(os.path.join(frame_root, rel_path[len("videos/"):]))
    if rel_path.startswith("images/"):
        candidates.append(os.path.join(frame_root, rel_path[len("images/"):]))
        candidates.append(os.path.join(frame_root, f"query_{rel_path}"))

    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _load_clip_frames(entry, frame_root, clip_root, num_video_frames):
    """Load clip frames from pre-extracted frame_path first, then fallback to mp4 extraction."""
    frame_path = _resolve_momentseeker_frame_path(frame_root, entry.get("frame_path"))
    if frame_path is not None:
        return [frame_path], frame_path

    output_path = entry.get("output_path", "")
    clip_name = output_path.replace("/", "_").split(".mp4")[0]
    cand_clip_frame_dir = os.path.join(frame_root, "video_frames", clip_name)
    if not os.path.exists(cand_clip_frame_dir):
        cand_clip_abs_path = os.path.join(clip_root, output_path)
        if not os.path.exists(cand_clip_abs_path):
            raise FileNotFoundError(
                f"[momentseeker] missing clip source: {cand_clip_abs_path}. "
                "Neither frame_path nor clip mp4 is available."
            )
        save_frames(video_path=cand_clip_abs_path, frame_dir=cand_clip_frame_dir, max_frames_saved=num_video_frames)
    clip_frames = load_frames(cand_clip_frame_dir)
    return clip_frames, cand_clip_frame_dir


@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    image_resolution = kwargs['image_resolution']
    ## metadata
    num_negative_clips = kwargs["num_negative_clips"]
    num_video_frames = kwargs["num_video_frames"]
    model_backbone = kwargs["model_backbone"]
    video_root, clip_root, frame_root = kwargs["video_root"], kwargs["clip_root"], kwargs["frame_root"]

    query_texts, query_images, cand_texts, cand_clip_images, dataset_infos = [], [], [], [], []
    for query, positive_frames, negative_frames, input_frames in \
            zip(batch_dict['query'], batch_dict["positive_frames"], batch_dict["negative_frames"], batch_dict["input_frames"]):

        if (input_frames.endswith(".mp4")):
            query_texts.append([process_input_text(TASK_INST_QRY_VIDEO, model_backbone, text=query, add_video_token=True)])
            query_video_name = input_frames.split(".mp4")[0].replace("/", "_")
            if query_video_name == 'movie101_77':  # TODO @yuepeng a buggy video?
                pass
            query_frame_dir = os.path.join(frame_root, "video_frames", query_video_name)
            if not os.path.exists(query_frame_dir):
                query_video_path = os.path.join(video_root, input_frames)
                if os.path.exists(query_video_path):
                    save_frames(video_path=query_video_path,
                                frame_dir=query_frame_dir,
                                max_frames_saved=num_video_frames)
            if os.path.exists(query_frame_dir):
                qry_frame_paths = load_frames(query_frame_dir)
            else:
                fallback_qry = _resolve_momentseeker_frame_path(
                    frame_root,
                    positive_frames[0].get("frame_path") if len(positive_frames) > 0 else None,
                )
                if fallback_qry is None:
                    raise FileNotFoundError(
                        f"[momentseeker] missing query source for input_frames={input_frames}. "
                        "Neither query mp4-derived frames nor fallback frame_path exists."
                    )
                qry_frame_paths = [fallback_qry]
            query_images.append([ImageVideoInstance(
                bytes=[None] * len(qry_frame_paths),
                paths=qry_frame_paths,
                resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)] * len(qry_frame_paths),
            ).to_dict()])
        elif (input_frames.endswith(".jpg")):
            query_texts.append([process_input_text(TASK_INST_QRY_IMG, model_backbone, text=query, add_image_token=True)])
            input_image_path = _resolve_momentseeker_frame_path(frame_root, input_frames)
            if input_image_path is None:
                input_image_path = _resolve_momentseeker_frame_path(
                    frame_root,
                    positive_frames[0].get("frame_path") if len(positive_frames) > 0 else None,
                )
            if input_image_path is None:
                raise FileNotFoundError(
                    f"[momentseeker] missing image query source for input_frames={input_frames}. "
                    "Checked query/input path variants and positive frame_path fallback."
                )
            query_images.append([ImageVideoInstance(
                bytes=[None],
                paths=[input_image_path],
                resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)],
            ).to_dict()])
        else:
            query_texts.append([process_input_text(TASK_INST_QRY_TEXT, model_backbone, text=query)])
            query_images.append([None])

        pos_clip_name, cand_clip_names, cand_frames = [], [], []
        for entry in positive_frames:
            pos_clip_frames, clip_name_for_info = _load_clip_frames(
                entry=entry,
                frame_root=frame_root,
                clip_root=clip_root,
                num_video_frames=num_video_frames,
            )
            cand_frames.append(ImageVideoInstance(
                bytes=[None] * len(pos_clip_frames),
                paths=pos_clip_frames,
                resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)] * len(pos_clip_frames),
            ).to_dict())
            cand_clip_names.append(clip_name_for_info)
            pos_clip_name.append(clip_name_for_info)
        for entry in negative_frames:
            neg_clip_frames, clip_name_for_info = _load_clip_frames(
                entry=entry,
                frame_root=frame_root,
                clip_root=clip_root,
                num_video_frames=num_video_frames,
            )
            cand_frames.append(ImageVideoInstance(
                bytes=[None] * len(neg_clip_frames),
                paths=neg_clip_frames,
                resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)] * len(neg_clip_frames),
            ).to_dict())
            cand_clip_names.append(clip_name_for_info)
        cand_texts.append([process_input_text(TASK_INST_TGT, model_backbone, add_video_token=True)] * (len(positive_frames) + len(negative_frames)))
        cand_clip_images.append(cand_frames)
        dataset_infos.append({
            "cand_names": cand_clip_names,
            "label_name": pos_clip_name,
        })

    return {"query_text": query_texts, "query_image": query_images,
            "cand_text": cand_texts, "cand_image": cand_clip_images,
            "dataset_infos": dataset_infos}


DATASET_PARSER_NAME = "momentseeker"
@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_momentseeker_dataset(model_args, data_args, *args, **kwargs):
    if kwargs.get("data_path", None) != None:
        dataset = load_dataset("json", data_files=kwargs["data_path"])
        dataset = dataset["train"]
    else:
        dataset = load_hf_dataset(EVAL_DATASET_HF_PATH[kwargs['dataset_name']])
    dataset = sample_dataset(dataset, **kwargs)

    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution

    dataset = dataset.map(lambda x: data_prepare(x, **kwargs), batched=True,
                          batch_size=2048, num_proc=1,
                          drop_last_batch=False, load_from_cache_file=False)
    dataset = dataset.select_columns(["query_text", "query_image", "cand_text", "cand_image", "dataset_infos"])
    return dataset, None
