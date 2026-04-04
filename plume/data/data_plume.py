import copy
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import BatchSampler, Dataset, Sampler
import transformers

from .rope2d import get_rope_index_2, get_rope_index_25

IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

THINK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
ANSWER_OPEN_RE = re.compile(r"<answer>", re.IGNORECASE)
ANSWER_CLOSE_RE = re.compile(r"</answer>", re.IGNORECASE)


def rank0_print(*args) -> None:
    if (
        (not torch.distributed.is_available())
        or (not torch.distributed.is_initialized())
        or torch.distributed.get_rank() == 0
    ):
        print(*args)


def _iter_jsonl_records(path: str) -> Iterator[Tuple[int, dict]]:
    with open(path, "r", encoding="utf-8") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.strip():
                continue
            yield offset, json.loads(line)


def _parse_curriculum_fractions(raw: str) -> List[float]:
    if not raw:
        return [1.0]
    values: List[float] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            value = float(piece)
        except ValueError:
            continue
        values.append(min(max(value, 0.0), 1.0))
    return values or [1.0]


def _parse_subset_filter(raw: str) -> set[str]:
    if not raw:
        return set()
    return {x.strip() for x in raw.split(",") if x.strip()}


def _iter_json_array_records(path: str) -> Iterator[dict]:
    decoder = json.JSONDecoder()
    chunk_size = 1 << 20
    with open(path, "r", encoding="utf-8") as f:
        buffer = ""
        started = False
        while True:
            if not buffer:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                buffer += chunk

            buffer = buffer.lstrip()
            if not buffer:
                continue

            if not started:
                if buffer[0] != "[":
                    raise ValueError(
                        f"Unsupported annotation format in {path}. Expected a top-level JSON array."
                    )
                buffer = buffer[1:]
                started = True
                continue

            buffer = buffer.lstrip()
            if not buffer:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                buffer += chunk
                continue

            if buffer[0] == "]":
                return
            if buffer[0] == ",":
                buffer = buffer[1:]
                continue

            while True:
                try:
                    record, end_idx = decoder.raw_decode(buffer)
                    buffer = buffer[end_idx:]
                    break
                except json.JSONDecodeError:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        raise
                    buffer += chunk

            if isinstance(record, dict):
                yield record


def _load_annotations(annotation_path: str) -> List[dict]:
    if annotation_path.endswith(".jsonl"):
        return [record for _, record in _iter_jsonl_records(annotation_path)]
    return list(_iter_json_array_records(annotation_path))


def _convert_json_array_to_jsonl(json_path: str, jsonl_path: str) -> str:
    rank0_print(f"[PLUME][DATA] convert json->jsonl: {json_path} -> {jsonl_path}")
    tmp_path = f"{jsonl_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for record in _iter_json_array_records(json_path):
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")
    os.replace(tmp_path, jsonl_path)
    return jsonl_path


def _ensure_jsonl_annotation_path(annotation_path: str) -> str:
    if annotation_path.endswith(".jsonl"):
        jsonl_path = annotation_path
        json_path = os.path.splitext(annotation_path)[0] + ".json"
    else:
        json_path = annotation_path
        jsonl_path = os.path.splitext(annotation_path)[0] + ".jsonl"

    if os.path.exists(jsonl_path):
        return jsonl_path
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    lock_path = f"{jsonl_path}.lock"
    while True:
        if os.path.exists(jsonl_path):
            return jsonl_path
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            try:
                return _convert_json_array_to_jsonl(json_path, jsonl_path)
            finally:
                if os.path.exists(lock_path):
                    os.remove(lock_path)
        except FileExistsError:
            rank0_print(f"[PLUME][DATA] waiting for jsonl conversion lock: {lock_path}")
            time.sleep(5)


def _get_side_dict_for_stats_from_sample(sample: dict, side_key: str) -> Optional[dict]:
    side = sample.get(side_key)
    if isinstance(side, dict):
        return side
    if isinstance(side, list) and len(side) == 1 and isinstance(side[0], dict):
        return side[0]
    return None


def _has_media_in_side(side: Optional[dict], image_key: str) -> bool:
    if not isinstance(side, dict):
        return False
    if image_key == "image":
        media = side.get("image", side.get("images", None))
    else:
        media = side.get("video", side.get("videos", None))
    if media is None:
        return False
    if isinstance(media, list):
        return len(media) > 0
    return True


def _build_jsonl_index(
    annotation_path: str,
    subset_filter: set[str],
) -> Tuple[List[int], List[str], Counter[str], Dict[str, List[int]], Dict[str, int]]:
    line_offsets: List[int] = []
    dataset_name_by_index: List[str] = []
    dataset_counter: Counter[str] = Counter()
    dataset_to_indices: Dict[str, List[int]] = defaultdict(list)
    stats = {
        "qry_image": 0,
        "qry_video": 0,
        "pos_image": 0,
        "pos_video": 0,
        "has_real_pair": 0,
    }

    for offset, record in _iter_jsonl_records(annotation_path):
        if not isinstance(record, dict):
            continue
        if not _annotation_matches_subset(record, subset_filter):
            continue
        dataset_name = _resolve_dataset_name(record)
        sample_idx = len(line_offsets)
        line_offsets.append(offset)
        dataset_name_by_index.append(dataset_name)
        dataset_counter[dataset_name] += 1
        dataset_to_indices[dataset_name].append(sample_idx)

        qry_side = _get_side_dict_for_stats_from_sample(record, "qry")
        pos_side = _get_side_dict_for_stats_from_sample(record, "pos")
        if qry_side is not None and pos_side is not None:
            stats["has_real_pair"] += 1
        if _has_media_in_side(qry_side, "image"):
            stats["qry_image"] += 1
        if _has_media_in_side(qry_side, "video"):
            stats["qry_video"] += 1
        if _has_media_in_side(pos_side, "image"):
            stats["pos_image"] += 1
        if _has_media_in_side(pos_side, "video"):
            stats["pos_video"] += 1

    return line_offsets, dataset_name_by_index, dataset_counter, dataset_to_indices, stats


def _first_existing_dict(item: dict, keys: Sequence[str]) -> Optional[dict]:
    for key in keys:
        value = item.get(key)
        if isinstance(value, dict):
            return value
    return None


def _collect_dataset_name_candidates(item: dict) -> List[str]:
    candidates: List[str] = []
    for key in ("dataset_name", "dataset", "subset", "source", "task_name"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())

    for side_key in ("qry", "pos"):
        side = item.get(side_key)
        if not isinstance(side, dict):
            continue
        for key in ("dataset_name", "dataset", "subset", "source", "task_name"):
            value = side.get(key)
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())

    return candidates


def _resolve_dataset_name(item: dict) -> str:
    candidates = _collect_dataset_name_candidates(item)
    if not candidates:
        return "<missing_dataset_name>"
    return candidates[0]


def _annotation_matches_subset(item: dict, subset_filter: set[str]) -> bool:
    if not subset_filter:
        return True

    candidates = _collect_dataset_name_candidates(item)
    if not candidates:
        # Be permissive when metadata is missing.
        return True

    return any(candidate in subset_filter for candidate in candidates)


def _extract_role(conv: dict) -> str:
    role = conv.get("role", conv.get("from", ""))
    if role == "human":
        return "user"
    if role == "gpt":
        return "assistant"
    return str(role)


def _extract_text(conv: dict) -> str:
    if "content" in conv:
        return str(conv["content"])
    return str(conv.get("value", ""))


def _set_text(conv: dict, text: str) -> None:
    if "content" in conv:
        conv["content"] = text
    else:
        conv["value"] = text


def _split_think_into_segments(think_text: str, num_segments: int) -> List[str]:
    """Split think text into *num_segments* roughly-equal segments by sentence.

    Heuristic: split on Chinese/English sentence-ending punctuation first,
    then distribute sentences into *num_segments* buckets.  If there are fewer
    sentences than segments, fall back to character-level chunking.
    """
    num_segments = max(int(num_segments), 1)
    think_text = think_text.strip()
    if not think_text:
        return [""] * num_segments

    # Sentence-level split (keep delimiters attached to the preceding sentence).
    sentence_re = re.compile(r"(?<=[。！？.!?\n])\s*")
    raw_sentences = [s for s in sentence_re.split(think_text) if s.strip()]

    if len(raw_sentences) >= num_segments:
        # Distribute sentences into num_segments buckets as evenly as possible.
        segments: List[str] = []
        base, extra = divmod(len(raw_sentences), num_segments)
        idx = 0
        for i in range(num_segments):
            count = base + (1 if i < extra else 0)
            segments.append("".join(raw_sentences[idx : idx + count]).strip())
            idx += count
        return segments

    # Fewer sentences than segments → character-level chunking.
    total_len = len(think_text)
    base_len, extra_chars = divmod(total_len, num_segments)
    segments = []
    pos = 0
    for i in range(num_segments):
        chunk_len = base_len + (1 if i < extra_chars else 0)
        segments.append(think_text[pos : pos + chunk_len])
        pos += chunk_len
    return segments


def _build_latent_assistant_text(
    raw_text: str,
    latent_ct_tokens: int,
    think_segments: int = 1,
    ct_per_segment: int = 1,
    drop_answer_text: bool = False,
) -> str:
    """Build the assistant turn with gradual latent replacement.

    The original CoT inside ``<think>`` is split into *think_segments* (S)
    segments.  The first ``k = latent_ct_tokens // ct_per_segment`` segments
    are replaced by ``<bot><ct>*(k*ct_per_segment)<eot>``, while the remaining
    ``S-k`` segments are kept as plain language reasoning text.
    """
    raw_text = raw_text or ""

    # --- extract think body & answer body ---
    think_match = THINK_RE.search(raw_text)
    think_body = think_match.group(1).strip() if think_match else ""

    answer_match = ANSWER_RE.search(raw_text)
    if answer_match:
        answer_body = answer_match.group(1).strip()
    else:
        stripped = THINK_RE.sub("", raw_text)
        stripped = ANSWER_OPEN_RE.sub("", stripped)
        stripped = ANSWER_CLOSE_RE.sub("", stripped)
        answer_body = stripped.strip()

    think_segments = max(int(think_segments), 1)
    ct_per_segment = max(int(ct_per_segment), 1)
    latent_ct_tokens = max(int(latent_ct_tokens), 0)

    def _format_answer(answer_text: str, drop_text: bool) -> str:
        if drop_text:
            return ""
        return f"<answer>{answer_text}</answer>"

    # Number of segments to replace with latent tokens.
    k = latent_ct_tokens // ct_per_segment
    k = min(k, think_segments)
    answer_section = _format_answer(answer_body, drop_answer_text)

    if k == 0:
        # Stage-0: no latent replacement, keep <bot><eot> markers for split.
        latent_block = "<bot><eot>"
        if think_body:
            rewritten = f"<think>{latent_block}{think_body}</think>{answer_section}"
        else:
            rewritten = f"<think>{latent_block}</think>{answer_section}"
    elif k >= think_segments or not think_body:
        # Full replacement (or no think text to keep).
        latent_block = "<bot>" + ("<ct>" * latent_ct_tokens) + "<eot>"
        rewritten = f"<think>{latent_block}</think>{answer_section}"
    else:
        # --- Gradual replacement: first k segments → latent, rest → text ---
        segments = _split_think_into_segments(think_body, think_segments)
        num_ct = k * ct_per_segment
        latent_block = "<bot>" + ("<ct>" * num_ct) + "<eot>"
        kept_text = "".join(segments[k:]).strip()
        if kept_text:
            rewritten = f"<think>{latent_block}{kept_text}</think>{answer_section}"
        else:
            rewritten = f"<think>{latent_block}</think>{answer_section}"

    rewritten = rewritten.strip()
    if "<gen_emb>" not in rewritten:
        rewritten = f"{rewritten}\n<gen_emb>" if rewritten else "<gen_emb>"
    return rewritten


def preprocess_qwen_2_visual(
    sources: List[List[dict]],
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw_image: Optional[List[int]] = None,
    grid_thw_video: Optional[List[int]] = None,
) -> Dict[str, torch.Tensor]:
    roles = {"human": "user", "gpt": "assistant"}
    system_message = "You are a helpful assistant."

    # Avoid copy.deepcopy(tokenizer) per sample — it's extremely slow.
    # Instead, temporarily override chat_template and restore after tokenization.
    _PLUME_CHAT_TEMPLATE = (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
    )
    _orig_chat_template = tokenizer.chat_template
    tokenizer.chat_template = _PLUME_CHAT_TEMPLATE

    image_idx = 0
    video_idx = 0
    input_ids: List[List[int]] = []
    labels: List[List[int]] = []

    for source in sources:
        if not source:
            continue

        first_role = _extract_role(source[0])
        if roles.get(first_role, first_role) != "user":
            source = source[1:]

        cur_input = tokenizer.apply_chat_template([{"role": "system", "content": system_message}])
        cur_labels = [IGNORE_INDEX] * len(cur_input)

        for conv in source:
            role = _extract_role(conv)
            role = roles.get(role, role)
            content = _extract_text(conv)

            if role == "user":
                if DEFAULT_IMAGE_TOKEN in content and grid_thw_image is not None:
                    parts = content.split(DEFAULT_IMAGE_TOKEN)
                    rebuilt: List[str] = []
                    for part_idx in range(len(parts) - 1):
                        rebuilt.append(parts[part_idx])
                        repeat = int(grid_thw_image[image_idx])
                        rebuilt.append(
                            "<|vision_start|>" + ("<|image_pad|>" * repeat) + "<|vision_end|>"
                        )
                        image_idx += 1
                    rebuilt.append(parts[-1])
                    content = "".join(rebuilt)

                if DEFAULT_VIDEO_TOKEN in content and grid_thw_video is not None:
                    parts = content.split(DEFAULT_VIDEO_TOKEN)
                    rebuilt = []
                    for part_idx in range(len(parts) - 1):
                        rebuilt.append(parts[part_idx])
                        repeat = int(grid_thw_video[video_idx])
                        rebuilt.append(
                            "<|vision_start|>" + ("<|video_pad|>" * repeat) + "<|vision_end|>"
                        )
                        video_idx += 1
                    rebuilt.append(parts[-1])
                    content = "".join(rebuilt)

            encoded = tokenizer.apply_chat_template([{"role": role, "content": content}])
            cur_input += encoded
            if role in {"user", "system"}:
                cur_labels += [IGNORE_INDEX] * len(encoded)
            else:
                # Ignore assistant role prefix tokens like "<|im_start|>assistant\n".
                masked = encoded.copy()
                prefix_len = min(3, len(masked))
                masked[:prefix_len] = [IGNORE_INDEX] * prefix_len
                cur_labels += masked

        assert len(cur_input) == len(cur_labels), f"{len(cur_input)} != {len(cur_labels)}"
        input_ids.append(cur_input)
        labels.append(cur_labels)

    if not input_ids:
        tokenizer.chat_template = _orig_chat_template
        raise ValueError("Empty conversation after preprocessing.")

    tokenizer.chat_template = _orig_chat_template
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


class CurriculumBalancedSubsetBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset: "LazyPlumeSFTDataset",
        batch_size: int,
        world_size: int,
        drop_last: bool,
        seed: int = 0,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.world_size = int(world_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)

    def _build_dataset_global_batches(
        self,
        dataset_name: str,
        indices: Sequence[int],
        rng: random.Random,
    ) -> List[List[List[int]]]:
        shuffled = list(indices)
        rng.shuffle(shuffled)
        global_batch_size = self.batch_size * self.world_size
        global_batches: List[List[List[int]]] = []

        for start in range(0, len(shuffled), global_batch_size):
            global_batch = shuffled[start : start + global_batch_size]
            if len(global_batch) < global_batch_size:
                if self.drop_last:
                    continue
                if not global_batch:
                    continue
                while len(global_batch) < global_batch_size:
                    global_batch.append(rng.choice(indices))

            local_batches = [
                global_batch[offset : offset + self.batch_size]
                for offset in range(0, global_batch_size, self.batch_size)
            ]
            if len(local_batches) == self.world_size and all(
                len(local_batch) == self.batch_size for local_batch in local_batches
            ):
                global_batches.append(local_batches)

        return global_batches

    def _spread_batch_slots(self, num_batches: int, total_slots: int) -> List[int]:
        if num_batches <= 0 or total_slots <= 0:
            return []
        if num_batches == 1:
            return [total_slots // 2]
        return [
            int((batch_idx * (total_slots - 1)) // (num_batches - 1))
            for batch_idx in range(num_batches)
        ]

    def _build_schedule(self) -> List[List[int]]:
        dataset = self.dataset
        if not dataset.dataset_to_indices:
            return []

        max_global_batches = max(dataset.dataset_global_batch_counts.values())
        if max_global_batches <= 0:
            return []

        epoch_seed = self.seed + int(getattr(dataset, "sampler_epoch", 0))
        slot_to_batches: Dict[int, List[Tuple[str, List[List[int]]]]] = defaultdict(list)
        for dataset_name in dataset.dataset_names:
            global_batches = dataset.dataset_global_batches.get(dataset_name, [])
            if not global_batches:
                continue
            for slot_idx, global_batch in zip(
                self._spread_batch_slots(len(global_batches), max_global_batches),
                global_batches,
            ):
                slot_to_batches[slot_idx].append((dataset_name, global_batch))

        schedule: List[List[int]] = []
        for batch_slot in range(max_global_batches):
            slot_batches = list(slot_to_batches.get(batch_slot, []))
            random.Random(epoch_seed + batch_slot * 104729).shuffle(slot_batches)
            for _, global_batch in slot_batches:
                schedule.extend(global_batch)
        return schedule

    def __iter__(self) -> Iterator[List[int]]:
        epoch_seed = self.seed + int(getattr(self.dataset, "sampler_epoch", 0))
        dataset_global_batches: Dict[str, List[List[List[int]]]] = {}
        dataset_global_batch_counts: Dict[str, int] = {}
        for offset, dataset_name in enumerate(self.dataset.dataset_names):
            indices = self.dataset.dataset_to_indices[dataset_name]
            rng = random.Random(epoch_seed + offset * 65537)
            global_batches = self._build_dataset_global_batches(dataset_name, indices, rng)
            if global_batches:
                dataset_global_batches[dataset_name] = global_batches
                dataset_global_batch_counts[dataset_name] = len(global_batches)

        self.dataset.dataset_global_batches = dataset_global_batches
        self.dataset.dataset_global_batch_counts = dataset_global_batch_counts
        for batch in self._build_schedule():
            yield batch

    def __len__(self) -> int:
        if not self.dataset.dataset_global_batch_counts:
            if not self.dataset.dataset_to_indices:
                return 0
            total_global_batches = 0
            denominator = self.batch_size * self.world_size
            for indices in self.dataset.dataset_to_indices.values():
                if self.drop_last:
                    total_global_batches += len(indices) // denominator
                else:
                    total_global_batches += math.ceil(len(indices) / denominator)
            return total_global_batches * self.world_size
        return sum(self.dataset.dataset_global_batch_counts.values()) * self.world_size


class LazyPlumeSFTDataset(Dataset):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args, model_args):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.model_args = model_args

        self.model_type = (
            "qwen2.5vl" if "qwen2.5" in model_args.model_name_or_path.lower() else "qwen2vl"
        )
        self.get_rope_index = get_rope_index_25 if self.model_type == "qwen2.5vl" else get_rope_index_2

        self.data_args.image_processor.max_pixels = data_args.max_pixels
        self.data_args.image_processor.min_pixels = data_args.min_pixels
        self.data_args.image_processor.size["longest_edge"] = data_args.max_pixels
        self.data_args.image_processor.size["shortest_edge"] = data_args.min_pixels

        self.curriculum_fractions = _parse_curriculum_fractions(data_args.plume_curriculum_stages)
        self.curriculum_stage = 0
        self.curriculum_progress = 0.0
        self.final_stage_portion = min(
            max(float(getattr(data_args, "plume_final_stage_portion", 0.5)), 0.0),
            1.0,
        )
        self.latent_answer_in_final_half = bool(
            getattr(data_args, "plume_latent_answer_in_final_half", False)
        )
        self.final_stage_answer_portion = min(
            max(float(getattr(data_args, "plume_final_stage_answer_portion", 0.5)), 0.0),
            1.0,
        )
        self.answer_latent_active = False
        self.use_qry = bool(data_args.plume_use_qry)
        self.use_pos = bool(data_args.plume_use_pos)
        self.include_gen_emb_loss = bool(data_args.plume_include_gen_emb_loss)
        self.media_root = str(data_args.plume_media_root or "")
        self.subset_filter_raw = str(data_args.plume_subset_filter or "")
        raw_max_seq_len = int(getattr(tokenizer, "model_max_length", 0) or 0)
        # HuggingFace tokenizer may use very large sentinel values when "no limit".11
        self.max_seq_length = (
            raw_max_seq_len if (raw_max_seq_len > 0 and raw_max_seq_len < 1_000_000) else 0
        )
        self._truncated_sample_count = 0
        self._truncated_token_count = 0

        think_segments = max(int(data_args.plume_think_segments), 1)
        ct_per_segment = max(int(data_args.plume_ct_tokens_per_segment), 1)
        self.max_latent_tokens = think_segments * ct_per_segment

        self.bot_token_id = tokenizer.convert_tokens_to_ids("<bot>")
        self.eot_token_id = tokenizer.convert_tokens_to_ids("<eot>")
        self.gen_emb_token_id = tokenizer.convert_tokens_to_ids("<gen_emb>")
        if self.bot_token_id < 0 or self.eot_token_id < 0 or self.gen_emb_token_id < 0:
            raise ValueError("Missing required special tokens <bot>/<eot>/<gen_emb> in tokenizer.")

        # Enable lazy tokenization for multi-worker scenarios.
        # When True, __getitem__ returns raw conversations + images (no tokenization).
        # Tokenization happens in collator (main process) with current curriculum_stage.
        self.use_lazy_tokenization = False  # Set by make_plume_data_module

        original_annotation_path = str(data_args.plume_annotation_path)
        self.annotation_path = _ensure_jsonl_annotation_path(original_annotation_path)
        self.data_args.plume_annotation_path = self.annotation_path
        self.annotation_is_jsonl = self.annotation_path.endswith(".jsonl")
        self._annotation_file = None
        self._annotation_file_pid: Optional[int] = None
        subset_filter = _parse_subset_filter(data_args.plume_subset_filter)
        (
            self.line_offsets,
            self.dataset_name_by_index,
            self.dataset_counter,
            self.dataset_to_indices,
            self.dataset_stats,
        ) = _build_jsonl_index(self.annotation_path, subset_filter)
        if not self.line_offsets:
            raise ValueError("No valid plume samples found after filtering.")

        self.sample_count = len(self.line_offsets)
        self.list_data_dict = None

        self.sampler_epoch = 0
        self.dataset_names = sorted(self.dataset_to_indices.keys())
        self.dataset_global_batches: Dict[str, List[List[List[int]]]] = {}
        self.dataset_global_batch_counts: Dict[str, int] = {}

        self._dataset_overview_cache = None
        self._log_dataset_overview()

    def __len__(self) -> int:
        return self.sample_count

    def num_curriculum_stages(self) -> int:
        return len(self.curriculum_fractions)

    def set_sampler_epoch(self, epoch: int) -> None:
        self.sampler_epoch = int(epoch)

    def curriculum_stage_for_progress(self, progress: float) -> int:
        if not self.curriculum_fractions:
            return 0
        progress = min(max(float(progress), 0.0), 0.999999)
        stage_count = len(self.curriculum_fractions)
        if stage_count <= 1:
            return 0

        final_stage_portion = min(max(float(self.final_stage_portion), 0.0), 1.0)
        if final_stage_portion >= 1.0:
            return stage_count - 1
        if final_stage_portion <= 0.0:
            return min(int(progress * stage_count), stage_count - 1)

        pre_final_count = stage_count - 1
        pre_final_portion = 1.0 - final_stage_portion
        if pre_final_count <= 0 or progress >= pre_final_portion:
            return stage_count - 1

        normalized = progress / max(pre_final_portion, 1e-8)
        return min(int(normalized * pre_final_count), pre_final_count - 1)

    def _final_stage_answer_active_for_progress(self, progress: float) -> bool:
        if not self.latent_answer_in_final_half:
            return False
        stage_count = len(self.curriculum_fractions)
        if stage_count <= 0:
            return False
        if self.curriculum_stage != stage_count - 1:
            return False
        final_stage_portion = min(max(float(self.final_stage_portion), 0.0), 1.0)
        if final_stage_portion <= 0.0:
            return False
        final_stage_start = 1.0 - final_stage_portion
        if progress < final_stage_start:
            return False
        local_progress = (progress - final_stage_start) / max(final_stage_portion, 1e-8)
        local_progress = min(max(local_progress, 0.0), 0.999999)
        answer_portion = min(max(float(self.final_stage_answer_portion), 0.0), 1.0)
        answer_start = 1.0 - answer_portion
        return local_progress >= answer_start

    def set_curriculum_stage(self, stage: int, progress: Optional[float] = None) -> None:
        if not self.curriculum_fractions:
            self.curriculum_stage = 0
            self.curriculum_progress = 0.0
            self.answer_latent_active = False
            return
        self.curriculum_stage = int(min(max(stage, 0), len(self.curriculum_fractions) - 1))
        if progress is not None:
            self.curriculum_progress = min(max(float(progress), 0.0), 0.999999)
        self.answer_latent_active = self._final_stage_answer_active_for_progress(self.curriculum_progress)

    def get_curriculum_state(self) -> Dict[str, float]:
        total = len(self.curriculum_fractions)
        stage = int(self.curriculum_stage if total > 0 else 0)
        ratio = float(self.curriculum_fractions[stage]) if total > 0 else 1.0
        return {
            "stage": float(stage),
            "total_stages": float(max(total, 1)),
            "replace_ratio": ratio,
            "latent_tokens": float(self._current_latent_tokens()),
            "max_latent_tokens": float(self.max_latent_tokens),
            "progress": float(self.curriculum_progress),
            "answer_latent_active": float(1.0 if self.answer_latent_active else 0.0),
        }

    def _get_side_dict_for_stats(self, sample: dict, side_key: str) -> Optional[dict]:
        return _get_side_dict_for_stats_from_sample(sample, side_key)

    def _has_media(self, side: Optional[dict], image_key: str) -> bool:
        return _has_media_in_side(side, image_key)

    def _load_sample(self, index: int) -> dict:
        if index < 0 or index >= self.sample_count:
            raise IndexError(f"Sample index out of range: {index}")
        current_pid = os.getpid()
        if self._annotation_file is None or self._annotation_file_pid != current_pid:
            if self._annotation_file is not None:
                try:
                    self._annotation_file.close()
                except Exception:
                    pass
            self._annotation_file = open(self.annotation_path, "r", encoding="utf-8")
            self._annotation_file_pid = current_pid
        self._annotation_file.seek(self.line_offsets[index])
        line = self._annotation_file.readline()
        if not line:
            raise ValueError(f"Empty annotation line at index={index}, offset={self.line_offsets[index]}")
        return json.loads(line)

    def _log_dataset_overview(self) -> None:
        total = self.sample_count
        top_datasets = ", ".join(
            [f"{name}:{count}" for name, count in self.dataset_counter.most_common(8)]
        )
        curriculum_desc = ",".join([f"{x:.2f}" for x in self.curriculum_fractions])

        rank0_print("[PLUME][DATA] ===== Dataset Overview =====")
        rank0_print(f"[PLUME][DATA] annotation_path={self.data_args.plume_annotation_path}")
        rank0_print(
            f"[PLUME][DATA] annotation_format={'jsonl-indexed' if self.annotation_is_jsonl else 'unknown'}, "
            f"subset_filter={self.subset_filter_raw or '<none>'}, filtered_samples={total}"
        )
        rank0_print(
            f"[PLUME][DATA] model_type={self.model_type}, media_root={self.media_root or '<none>'}"
        )
        rank0_print(
            f"[PLUME][DATA] tokenizer_max_seq_len="
            f"{self.max_seq_length if self.max_seq_length > 0 else '<unlimited>'}"
        )
        rank0_print(
            f"[PLUME][DATA] use_qry={self.use_qry}, use_pos={self.use_pos}, "
            f"include_gen_emb_loss={self.include_gen_emb_loss}"
        )
        rank0_print(
            f"[PLUME][DATA] think_segments={int(self.max_latent_tokens // max(int(self.data_args.plume_ct_tokens_per_segment), 1))}, "
            f"ct_per_segment={int(self.data_args.plume_ct_tokens_per_segment)}, "
            f"max_latent_tokens={self.max_latent_tokens}"
        )
        rank0_print(
            f"[PLUME][DATA] curriculum_stages={len(self.curriculum_fractions)} "
            f"[{curriculum_desc}], final_stage_portion={self.final_stage_portion:.2f}, "
            f"latent_answer_in_final_half={self.latent_answer_in_final_half}, "
            f"final_stage_answer_portion={self.final_stage_answer_portion:.2f}"
        )
        rank0_print(
            f"[PLUME][DATA] real_qry_pos_pairs={self.dataset_stats['has_real_pair']}/{total}, "
            f"qry(image/video)={self.dataset_stats['qry_image']}/{self.dataset_stats['qry_video']}, "
            f"pos(image/video)={self.dataset_stats['pos_image']}/{self.dataset_stats['pos_video']}"
        )
        rank0_print(f"[PLUME][DATA] dataset_name_top={top_datasets}")
        rank0_print("[PLUME][DATA] ============================")

    def _current_replace_ratio(self) -> float:
        return float(self.curriculum_fractions[self.curriculum_stage])

    def _current_latent_tokens(self) -> int:
        ratio = self._current_replace_ratio()
        latent_tokens = int(round(self.max_latent_tokens * ratio))
        if ratio > 0.0 and latent_tokens == 0:
            latent_tokens = 1
        return max(latent_tokens, 0)
        # return 0

    def _truncate_sequence_if_needed(
        self,
        seq_ids: torch.Tensor,
        seq_labels: torch.Tensor,
        seq_pos: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.max_seq_length <= 0:
            return seq_ids, seq_labels, seq_pos

        seq_len = int(seq_ids.shape[0])
        if seq_len <= self.max_seq_length:
            return seq_ids, seq_labels, seq_pos

        cut = seq_len - self.max_seq_length

        # --- Smart truncation: keep head (vision tokens) + tail (special tokens),
        #     remove from the middle (CoT text) to preserve visual features and
        #     critical end tokens (<gen_emb>, <disc_emb>, etc.). ---
        VISION_END_ID = 151653  # <|vision_end|>
        vision_end_positions = torch.nonzero(seq_ids == VISION_END_ID, as_tuple=False).flatten()

        if vision_end_positions.numel() > 0:
            safe_head = int(vision_end_positions[-1].item()) + 1
        else:
            safe_head = 0

        safe_tail = 256
        available_middle = seq_len - safe_head - safe_tail

        if available_middle >= cut and safe_head > 0:
            # Middle truncation: remove `cut` tokens after the visual prefix
            mid_start = safe_head
            mid_end = mid_start + cut
            seq_ids = torch.cat([seq_ids[:mid_start], seq_ids[mid_end:]])
            seq_labels = torch.cat([seq_labels[:mid_start], seq_labels[mid_end:]])
            seq_pos = torch.cat([seq_pos[:, :mid_start], seq_pos[:, mid_end:]], dim=1)
        else:
            # Fallback: left-truncation
            seq_ids = seq_ids[cut:]
            seq_labels = seq_labels[cut:]
            seq_pos = seq_pos[:, cut:]

        self._truncated_sample_count += 1
        self._truncated_token_count += cut
        if self._truncated_sample_count <= 3 or self._truncated_sample_count % 500 == 0:
            rank0_print(
                f"[PLUME][DATA] truncate sample: original_len={seq_len}, "
                f"kept_len={self.max_seq_length}, dropped={cut}, "
                f"truncated_samples={self._truncated_sample_count}, "
                f"truncated_tokens={self._truncated_token_count}"
            )
        return seq_ids, seq_labels, seq_pos

    def _resolve_media_path(self, media_path: str, sample: dict, side: dict) -> str:
        media_path = str(media_path).strip()
        if not media_path:
            return media_path

        rel_variants: List[str] = []
        seen_rel: set[str] = set()
        for candidate in (media_path, media_path.lstrip("./")):
            if candidate and candidate not in seen_rel:
                rel_variants.append(candidate)
                seen_rel.add(candidate)

        # Be tolerant to occasional extension typos like ".jpgg".
        for rel in list(rel_variants):
            lower_rel = rel.lower()
            if lower_rel.endswith(".jpgg") or lower_rel.endswith(".jpegg"):
                fixed = rel[:-1]
                if fixed not in seen_rel:
                    rel_variants.append(fixed)
                    seen_rel.add(fixed)

        if media_path.startswith("images/"):
            mmeb_variant = os.path.join("MMEB-train", media_path)
            if mmeb_variant not in seen_rel:
                rel_variants.append(mmeb_variant)
                seen_rel.add(mmeb_variant)

        candidate_roots: List[str] = []
        if self.media_root:
            candidate_roots.append(self.media_root)
        for key in ("data_path", "media_root", "root"):
            value = side.get(key)
            if isinstance(value, str) and value:
                candidate_roots.append(value)
            value = sample.get(key)
            if isinstance(value, str) and value:
                candidate_roots.append(value)

        # Also try parent dirs, which helps when a root is too deep.
        expanded_roots: List[str] = []
        seen_root: set[str] = set()
        for root in candidate_roots:
            norm_root = os.path.normpath(root)
            if norm_root not in seen_root:
                expanded_roots.append(norm_root)
                seen_root.add(norm_root)
            parent = os.path.dirname(norm_root)
            if parent and parent not in seen_root:
                expanded_roots.append(parent)
                seen_root.add(parent)

        probe_paths: List[str] = []
        if os.path.isabs(media_path):
            probe_paths.extend(rel_variants)
        else:
            for rel in rel_variants:
                for root in expanded_roots:
                    probe_paths.append(os.path.normpath(os.path.join(root, rel)))
                probe_paths.append(os.path.normpath(rel))

        seen_probe: set[str] = set()
        for candidate in probe_paths:
            if candidate in seen_probe:
                continue
            seen_probe.add(candidate)
            if os.path.exists(candidate):
                return candidate

        # Best-effort fallback for clearer downstream errors.
        if not os.path.isabs(media_path) and self.media_root:
            return os.path.normpath(os.path.join(self.media_root, rel_variants[0]))
        if probe_paths:
            return probe_paths[0]
        return media_path

    def process_image_unified(self, image_file: str) -> Tuple[torch.Tensor, torch.Tensor]:
        processor = copy.deepcopy(self.data_args.image_processor)
        image = Image.open(image_file).convert("RGB")
        height, width = image.size
        if height < 28 or width < 28:
            scale_factor = max(28 / max(height, 1), 28 / max(width, 1))
            image = image.resize((int(width * scale_factor), int(height * scale_factor)))
        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, list):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw

    def process_video(self, frame_files: List[str]) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
        frames = [np.array(Image.open(f).convert("RGB")) for f in frame_files]
        processor = copy.deepcopy(self.data_args.image_processor)
        processor.max_pixels = self.data_args.video_max_frame_pixels
        processor.min_pixels = self.data_args.video_min_frame_pixels
        processor.size["longest_edge"] = processor.max_pixels
        processor.size["shortest_edge"] = processor.min_pixels

        fps = 2.0
        video_processed = processor.preprocess(images=None, videos=frames, return_tensors="pt")
        video_tensor = video_processed["pixel_values_videos"]
        grid_thw = video_processed["video_grid_thw"][0]
        second_per_grid_ts = [
            self.data_args.image_processor.temporal_patch_size / fps
        ] * int(len(grid_thw))
        return video_tensor, grid_thw, second_per_grid_ts

    def _choose_side(self, sample: dict, side_key: str, fallback_key: str) -> dict:
        side = sample.get(side_key)
        if isinstance(side, dict):
            return side
        if isinstance(side, list) and len(side) == 1 and isinstance(side[0], dict):
            return side[0]
        fallback = sample.get(fallback_key)
        if isinstance(fallback, dict):
            return fallback
        if isinstance(fallback, list) and len(fallback) == 1 and isinstance(fallback[0], dict):
            return fallback[0]
        raise ValueError(f"Sample missing both `{side_key}` and `{fallback_key}`.")

    def _rewrite_conversations(self, conversations: List[dict]) -> Tuple[List[dict], int]:
        conversations = copy.deepcopy(conversations)
        assistant_indices = [
            idx for idx, conv in enumerate(conversations) if _extract_role(conv) == "assistant"
        ]
        if not assistant_indices:
            raise ValueError("Conversation has no assistant turn.")

        latent_tokens = self._current_latent_tokens()
        target_idx = assistant_indices[-1]
        target_conv = conversations[target_idx]
        rewritten = _build_latent_assistant_text(
            _extract_text(target_conv),
            latent_tokens,
            think_segments=max(int(self.data_args.plume_think_segments), 1),
            ct_per_segment=max(int(self.data_args.plume_ct_tokens_per_segment), 1),
            drop_answer_text=bool(self.answer_latent_active),
        )
        _set_text(target_conv, rewritten)
        return conversations, latent_tokens

    def _build_side(self, sample: dict, side: dict) -> Dict[str, torch.Tensor]:
        image_tensors: Optional[torch.Tensor] = None
        image_grid_thw: Optional[torch.Tensor] = None
        video_tensors: Optional[torch.Tensor] = None
        video_grid_thw: Optional[torch.Tensor] = None
        second_per_grid_ts: Optional[List[float]] = None
        grid_thw_merged: Optional[List[int]] = None
        video_grid_thw_merged: Optional[List[int]] = None

        image_files = side.get("image", side.get("images", None))
        if image_files is not None:
            if not isinstance(image_files, list):
                image_files = [image_files]
            resolved = [self._resolve_media_path(str(x), sample, side) for x in image_files]
            existing_resolved = [path for path in resolved if os.path.exists(path)]
            if not existing_resolved:
                raise FileNotFoundError(
                    f"No valid image files found. raw={image_files}, resolved={resolved}"
                )
            processed = [self.process_image_unified(path) for path in existing_resolved]
            if processed:
                images, grid_list = zip(*processed)
                image_tensors = torch.cat(list(images), dim=0)
                image_grid_thw = torch.stack(list(grid_list), dim=0)
                merge_size = int(self.data_args.image_processor.merge_size)
                grid_thw_merged = [
                    int((thw.prod().item()) // (merge_size**2)) for thw in grid_list
                ]

        video_files = side.get("video", side.get("videos", None))
        if video_files is not None:
            if isinstance(video_files, list):
                resolved_video_files = [
                    self._resolve_media_path(str(x), sample, side) for x in video_files
                ]
            else:
                resolved_video_files = [
                    self._resolve_media_path(str(video_files), sample, side)
                ]
            existing_video_files = [
                path for path in resolved_video_files if os.path.exists(path)
            ]
            if existing_video_files:
                video_tensor, v_grid, second_per_grid_ts = self.process_video(existing_video_files)
                video_tensors = video_tensor
                video_grid_thw = v_grid.unsqueeze(0)
                merge_size = int(self.data_args.image_processor.merge_size)
                video_grid_thw_merged = [int((v_grid.prod().item()) // (merge_size**2))]

        conversations = side.get("conversations", None)
        if not isinstance(conversations, list):
            raise ValueError("Side record does not contain conversations list.")
        rewritten_convs, configured_latent_tokens = self._rewrite_conversations(conversations)
        if image_tensors is None or video_tensors is None:
            for conv in rewritten_convs:
                if _extract_role(conv) != "user":
                    continue
                text = _extract_text(conv)
                if image_tensors is None:
                    text = text.replace(DEFAULT_IMAGE_TOKEN, "")
                if video_tensors is None:
                    text = text.replace(DEFAULT_VIDEO_TOKEN, "")
                _set_text(conv, text)

        data_dict = preprocess_qwen_2_visual(
            [rewritten_convs],
            self.tokenizer,
            grid_thw_image=grid_thw_merged,
            grid_thw_video=video_grid_thw_merged,
        )
        input_ids = data_dict["input_ids"]  # [1, L]
        labels = data_dict["labels"]  # [1, L]

        if image_grid_thw is not None or video_grid_thw is not None:
            position_ids, _ = self.get_rope_index(
                self.data_args.image_processor.merge_size,
                input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
            )
        else:
            seq_len = input_ids.size(1)
            position_ids = (
                torch.arange(seq_len, dtype=torch.long).view(1, 1, -1).expand(3, 1, -1)
            )

        seq_ids = input_ids[0]
        seq_labels = labels[0]
        seq_pos = position_ids[:, 0, :]
        seq_ids, seq_labels, seq_pos = self._truncate_sequence_if_needed(
            seq_ids, seq_labels, seq_pos
        )

        # --- Guard: if left-truncation removed image/video pad tokens,
        #     clear the corresponding pixel_values to avoid the Qwen2-VL
        #     "Image features and image tokens do not match" error. --------
        IMAGE_PAD_ID = 151655   # <|image_pad|>
        VIDEO_PAD_ID = 151656   # <|video_pad|>
        remaining_img_tokens = int((seq_ids == IMAGE_PAD_ID).sum().item())
        remaining_vid_tokens = int((seq_ids == VIDEO_PAD_ID).sum().item())
        if image_tensors is not None:
            # NOTE: tokenization expands <image> using grid_thw_merged (= prod(thw) // merge_size^2),
            # so we must compare against the merged count, NOT the raw prod(thw).
            merge_size_sq = int(self.data_args.image_processor.merge_size) ** 2
            expected_img_tokens = int(image_grid_thw.prod(dim=1).sum().item()) // merge_size_sq if image_grid_thw is not None else 0
            if remaining_img_tokens == 0 or remaining_img_tokens != expected_img_tokens:
                image_tensors = None
                image_grid_thw = None
        if video_tensors is not None:
            merge_size_sq_v = int(self.data_args.image_processor.merge_size) ** 2
            expected_vid_tokens = int(video_grid_thw.prod(dim=1).sum().item()) // merge_size_sq_v if video_grid_thw is not None else 0
            if remaining_vid_tokens == 0 or remaining_vid_tokens != expected_vid_tokens:
                video_tensors = None
                video_grid_thw = None

        bot_positions = torch.nonzero(seq_ids == self.bot_token_id, as_tuple=False).flatten()
        eot_positions = torch.nonzero(seq_ids == self.eot_token_id, as_tuple=False).flatten()
        if bot_positions.numel() == 0 or eot_positions.numel() == 0:
            raise ValueError(
                "Cannot find <bot>/<eot> in tokenized sequence "
                f"(seq_len={int(seq_ids.shape[0])}, max_seq_len={self.max_seq_length})."
            )

        bot_pos = int(bot_positions[-1].item())
        eot_after_bot = eot_positions[eot_positions > bot_pos]
        if eot_after_bot.numel() == 0:
            raise ValueError("Cannot find <eot> after <bot>.")
        eot_pos = int(eot_after_bot[0].item())

        latent_steps = int(max(eot_pos - bot_pos - 1, 0))
        if configured_latent_tokens != latent_steps:
            # Keep runtime latent steps aligned with tokenized sequence.
            configured_latent_tokens = latent_steps

        prefix_ids = seq_ids[: bot_pos + 1]
        prefix_pos = seq_pos[:, : bot_pos + 1]

        suffix_ids = seq_ids[eot_pos:]
        suffix_pos = seq_pos[:, eot_pos:]
        suffix_labels = seq_labels[eot_pos:].clone()

        if not self.include_gen_emb_loss:
            suffix_labels[suffix_ids == self.gen_emb_token_id] = IGNORE_INDEX

        return {
            "prefix_input_ids": prefix_ids,
            "prefix_attention_mask": torch.ones_like(prefix_ids, dtype=torch.long),
            "prefix_position_ids": prefix_pos.long(),
            "suffix_input_ids": suffix_ids,
            "suffix_attention_mask": torch.ones_like(suffix_ids, dtype=torch.long),
            "suffix_position_ids": suffix_pos.long(),
            "suffix_labels": suffix_labels.long(),
            "plume_latent_steps": torch.tensor(configured_latent_tokens, dtype=torch.long),
            "pixel_values": image_tensors,
            "image_grid_thw": image_grid_thw,
            "pixel_values_videos": video_tensors,
            "video_grid_thw": video_grid_thw,
        }

    def _build_side_raw(self, sample: dict, side: dict) -> Dict[str, object]:
        """Build side data WITHOUT tokenization (for lazy tokenization in collator).

        Returns raw conversations and processed visual tensors.
        Workers call this method to avoid tokenizing with stale curriculum_stage.
        """
        image_tensors: Optional[torch.Tensor] = None
        image_grid_thw: Optional[torch.Tensor] = None
        video_tensors: Optional[torch.Tensor] = None
        video_grid_thw: Optional[torch.Tensor] = None
        second_per_grid_ts: Optional[List[float]] = None
        grid_thw_merged: Optional[List[int]] = None
        video_grid_thw_merged: Optional[List[int]] = None

        # Process images (same as _build_side lines 719-737)
        image_files = side.get("image", side.get("images", None))
        if image_files is not None:
            if not isinstance(image_files, list):
                image_files = [image_files]
            resolved = [self._resolve_media_path(str(x), sample, side) for x in image_files]
            existing_resolved = [path for path in resolved if os.path.exists(path)]
            if not existing_resolved:
                raise FileNotFoundError(
                    f"No valid image files found. raw={image_files}, resolved={resolved}"
                )
            processed = [self.process_image_unified(path) for path in existing_resolved]
            if processed:
                images, grid_list = zip(*processed)
                image_tensors = torch.cat(list(images), dim=0)
                image_grid_thw = torch.stack(list(grid_list), dim=0)
                merge_size = int(self.data_args.image_processor.merge_size)
                grid_thw_merged = [
                    int((thw.prod().item()) // (merge_size**2)) for thw in grid_list
                ]

        # Process videos (same as _build_side lines 739-757)
        video_files = side.get("video", side.get("videos", None))
        if video_files is not None:
            if isinstance(video_files, list):
                resolved_video_files = [
                    self._resolve_media_path(str(x), sample, side) for x in video_files
                ]
            else:
                resolved_video_files = [
                    self._resolve_media_path(str(video_files), sample, side)
                ]
            existing_video_files = [
                path for path in resolved_video_files if os.path.exists(path)
            ]
            if existing_video_files:
                video_tensor, v_grid, second_per_grid_ts = self.process_video(existing_video_files)
                video_tensors = video_tensor
                video_grid_thw = v_grid.unsqueeze(0)
                merge_size = int(self.data_args.image_processor.merge_size)
                video_grid_thw_merged = [int((v_grid.prod().item()) // (merge_size**2))]

        # Get conversations and clean up missing media tokens (same as _build_side lines 759-772)
        conversations = side.get("conversations", None)
        if not isinstance(conversations, list):
            raise ValueError("Side record does not contain conversations list.")

        # Deep copy conversations to avoid modifying original data
        conversations_cleaned = copy.deepcopy(conversations)

        # Remove image/video tokens if corresponding media is missing
        if image_tensors is None or video_tensors is None:
            for conv in conversations_cleaned:
                if _extract_role(conv) != "user":
                    continue
                text = _extract_text(conv)
                if image_tensors is None:
                    text = text.replace(DEFAULT_IMAGE_TOKEN, "")
                if video_tensors is None:
                    text = text.replace(DEFAULT_VIDEO_TOKEN, "")
                _set_text(conv, text)

        # Return raw data WITHOUT tokenization
        return {
            "conversations": conversations_cleaned,
            "pixel_values": image_tensors,
            "image_grid_thw": image_grid_thw,
            "grid_thw_merged": grid_thw_merged,
            "pixel_values_videos": video_tensors,
            "video_grid_thw": video_grid_thw,
            "video_grid_thw_merged": video_grid_thw_merged,
            "second_per_grid_ts": second_per_grid_ts,
        }

    def _get_item(self, index: int):
        """Get item with optional lazy tokenization mode.

        When use_lazy_tokenization=True (multi-worker mode):
          - Returns raw conversations + processed images/videos
          - Tokenization deferred to collator (main process)

        When use_lazy_tokenization=False (single-worker mode):
          - Returns fully tokenized tensors
          - Backward compatible with original behavior
        """
        sample = self._load_sample(index)
        qry_side = (
            self._choose_side(sample, "qry", "pos")
            if self.use_qry
            else self._choose_side(sample, "pos", "qry")
        )
        pos_side = (
            self._choose_side(sample, "pos", "qry")
            if self.use_pos
            else self._choose_side(sample, "qry", "pos")
        )

        if self.use_lazy_tokenization:
            # Lazy mode: return raw data for tokenization in collator
            qry = self._build_side_raw(sample, qry_side)
            pos = self._build_side_raw(sample, pos_side)
        else:
            # Eager mode: tokenize in worker (original behavior)
            qry = self._build_side(sample, qry_side)
            pos = self._build_side(sample, pos_side)

        real_pair = bool(isinstance(sample.get("qry"), dict) and isinstance(sample.get("pos"), dict))
        return qry, pos, real_pair

    def __getitem__(self, index: int):
        num_base_retries = 3
        num_fallback_retries = 8

        for try_idx in range(num_base_retries):
            try:
                return self._get_item(index)
            except Exception as e:
                print(f"[PLUME][Try {try_idx}] failed to fetch sample {index}: {e}")
                time.sleep(1)

        for try_idx in range(num_fallback_retries):
            try:
                rand_idx = random.randint(0, self.sample_count - 1)
                return self._get_item(rand_idx)
            except Exception as e:
                print(f"[PLUME][Fallback {try_idx}] failed random sample: {e}")

        return self._get_item(index)


def _pad_position_ids(pos_list: Sequence[torch.Tensor], max_len: int) -> torch.Tensor:
    # Output shape: [B, 3, max_len]
    batch_size = len(pos_list)
    out = torch.zeros((batch_size, 3, max_len), dtype=torch.long)
    for idx, pos in enumerate(pos_list):
        if pos.ndim != 2 or pos.shape[0] != 3:
            raise ValueError(f"Invalid position_ids shape: {tuple(pos.shape)}")
        cur_len = int(pos.shape[1])
        out[idx, :, :cur_len] = pos
        if cur_len < max_len and cur_len > 0:
            out[idx, :, cur_len:] = pos[:, -1:].expand(3, max_len - cur_len)
    return out


@dataclass
@dataclass
class DataCollatorForPlumeDataset:
    tokenizer: transformers.PreTrainedTokenizer
    dataset: Optional['LazyPlumeSFTDataset'] = None  # For accessing curriculum_stage during tokenization
    _logged_mode: bool = False  # Track if we've logged the tokenization mode

    def _tokenize_side_instance(self, raw_instance: Dict[str, object]) -> Dict[str, torch.Tensor]:
        """Tokenize a raw side instance with current curriculum stage.

        This runs in the main process with access to current curriculum_stage.
        Replicates the tokenization logic from _build_side() but uses the
        dataset's current curriculum_stage value.
        """
        if self.dataset is None:
            raise ValueError("Dataset reference required for lazy tokenization")

        # Extract raw data from instance
        conversations = raw_instance["conversations"]
        pixel_values = raw_instance.get("pixel_values", None)
        image_grid_thw = raw_instance.get("image_grid_thw", None)
        grid_thw_merged = raw_instance.get("grid_thw_merged", None)
        pixel_values_videos = raw_instance.get("pixel_values_videos", None)
        video_grid_thw = raw_instance.get("video_grid_thw", None)
        video_grid_thw_merged = raw_instance.get("video_grid_thw_merged", None)
        second_per_grid_ts = raw_instance.get("second_per_grid_ts", None)

        # Rewrite conversations with current curriculum stage (main process value)
        rewritten_convs, configured_latent_tokens = self.dataset._rewrite_conversations(conversations)

        # Tokenize conversations
        data_dict = preprocess_qwen_2_visual(
            [rewritten_convs],
            self.tokenizer,
            grid_thw_image=grid_thw_merged,
            grid_thw_video=video_grid_thw_merged,
        )
        input_ids = data_dict["input_ids"]  # [1, L]
        labels = data_dict["labels"]  # [1, L]

        # Compute position_ids
        if image_grid_thw is not None or video_grid_thw is not None:
            position_ids, _ = self.dataset.get_rope_index(
                self.dataset.data_args.image_processor.merge_size,
                input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
            )
        else:
            seq_len = input_ids.size(1)
            position_ids = (
                torch.arange(seq_len, dtype=torch.long).view(1, 1, -1).expand(3, 1, -1)
            )

        seq_ids = input_ids[0]
        seq_labels = labels[0]
        seq_pos = position_ids[:, 0, :]
        seq_ids, seq_labels, seq_pos = self.dataset._truncate_sequence_if_needed(
            seq_ids, seq_labels, seq_pos
        )

        # Guard: if left-truncation removed image/video pad tokens, clear corresponding pixel_values
        IMAGE_PAD_ID = 151655   # <|image_pad|>
        VIDEO_PAD_ID = 151656   # <|video_pad|>
        remaining_img_tokens = int((seq_ids == IMAGE_PAD_ID).sum().item())
        remaining_vid_tokens = int((seq_ids == VIDEO_PAD_ID).sum().item())
        if pixel_values is not None:
            merge_size_sq = int(self.dataset.data_args.image_processor.merge_size) ** 2
            expected_img_tokens = int(image_grid_thw.prod(dim=1).sum().item()) // merge_size_sq if image_grid_thw is not None else 0
            if remaining_img_tokens == 0 or remaining_img_tokens != expected_img_tokens:
                pixel_values = None
                image_grid_thw = None
        if pixel_values_videos is not None:
            merge_size_sq_v = int(self.dataset.data_args.image_processor.merge_size) ** 2
            expected_vid_tokens = int(video_grid_thw.prod(dim=1).sum().item()) // merge_size_sq_v if video_grid_thw is not None else 0
            if remaining_vid_tokens == 0 or remaining_vid_tokens != expected_vid_tokens:
                pixel_values_videos = None
                video_grid_thw = None

        # Find <bot> and <eot> positions
        bot_positions = torch.nonzero(seq_ids == self.dataset.bot_token_id, as_tuple=False).flatten()
        eot_positions = torch.nonzero(seq_ids == self.dataset.eot_token_id, as_tuple=False).flatten()
        if bot_positions.numel() == 0 or eot_positions.numel() == 0:
            raise ValueError(
                "Cannot find <bot>/<eot> in tokenized sequence "
                f"(seq_len={int(seq_ids.shape[0])}, max_seq_len={self.dataset.max_seq_length})."
            )

        bot_pos = int(bot_positions[-1].item())
        eot_after_bot = eot_positions[eot_positions > bot_pos]
        if eot_after_bot.numel() == 0:
            raise ValueError("Cannot find <eot> after <bot>.")
        eot_pos = int(eot_after_bot[0].item())

        latent_steps = int(max(eot_pos - bot_pos - 1, 0))
        if configured_latent_tokens != latent_steps:
            configured_latent_tokens = latent_steps

        prefix_ids = seq_ids[: bot_pos + 1]
        prefix_pos = seq_pos[:, : bot_pos + 1]

        suffix_ids = seq_ids[eot_pos:]
        suffix_pos = seq_pos[:, eot_pos:]
        suffix_labels = seq_labels[eot_pos:].clone()

        if not self.dataset.include_gen_emb_loss:
            suffix_labels[suffix_ids == self.dataset.gen_emb_token_id] = IGNORE_INDEX

        return {
            "prefix_input_ids": prefix_ids,
            "prefix_attention_mask": torch.ones_like(prefix_ids, dtype=torch.long),
            "prefix_position_ids": prefix_pos.long(),
            "suffix_input_ids": suffix_ids,
            "suffix_attention_mask": torch.ones_like(suffix_ids, dtype=torch.long),
            "suffix_position_ids": suffix_pos.long(),
            "suffix_labels": suffix_labels.long(),
            "plume_latent_steps": torch.tensor(configured_latent_tokens, dtype=torch.long),
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
        }

    def _collate_side(self, side_instances: Sequence[Dict[str, object]]) -> Dict[str, object]:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("Tokenizer must define pad_token_id for plume collator.")

        # Detect if instances are raw (lazy tokenization mode) and tokenize if needed
        if len(side_instances) > 0 and "conversations" in side_instances[0]:
            # Raw instances detected - tokenize in main process with current curriculum_stage
            if not self._logged_mode:
                print(f"[PLUME][COLLATOR] Using LAZY tokenization mode (tokenizing in main process)")
                self._logged_mode = True

            tokenized_instances = []
            for i, raw_inst in enumerate(side_instances):
                try:
                    tokenized = self._tokenize_side_instance(raw_inst)
                    tokenized_instances.append(tokenized)
                except Exception as e:
                    # Log error and skip failed sample
                    print(f"Warning: Failed to tokenize instance {i}: {e}")
                    continue

            if len(tokenized_instances) == 0:
                raise ValueError("All instances failed tokenization in collator")

            side_instances = tokenized_instances
        else:
            # Pre-tokenized instances (eager mode)
            if not self._logged_mode:
                print(f"[PLUME][COLLATOR] Using EAGER tokenization mode (pre-tokenized in workers)")
                self._logged_mode = True

        prefix_ids_list = [x["prefix_input_ids"] for x in side_instances]
        suffix_ids_list = [x["suffix_input_ids"] for x in side_instances]
        prefix_pos_list = [x["prefix_position_ids"] for x in side_instances]
        suffix_pos_list = [x["suffix_position_ids"] for x in side_instances]
        suffix_labels_list = [x["suffix_labels"] for x in side_instances]

        prefix_input_ids = torch.nn.utils.rnn.pad_sequence(
            prefix_ids_list, batch_first=True, padding_value=pad_id
        )
        suffix_input_ids = torch.nn.utils.rnn.pad_sequence(
            suffix_ids_list, batch_first=True, padding_value=pad_id
        )
        suffix_labels = torch.nn.utils.rnn.pad_sequence(
            suffix_labels_list, batch_first=True, padding_value=IGNORE_INDEX
        )

        prefix_max_len = int(prefix_input_ids.shape[1])
        suffix_max_len = int(suffix_input_ids.shape[1])
        prefix_position_ids = _pad_position_ids(prefix_pos_list, prefix_max_len)
        suffix_position_ids = _pad_position_ids(suffix_pos_list, suffix_max_len)

        batch = {
            "prefix_input_ids": prefix_input_ids,
            "prefix_attention_mask": prefix_input_ids.ne(pad_id).long(),
            "prefix_position_ids": prefix_position_ids,
            "suffix_input_ids": suffix_input_ids,
            "suffix_attention_mask": suffix_input_ids.ne(pad_id).long(),
            "suffix_position_ids": suffix_position_ids,
            "suffix_labels": suffix_labels,
            "plume_latent_steps": torch.stack(
                [x["plume_latent_steps"] for x in side_instances], dim=0
            ),
            # Keep visual tensors as per-sample lists to match PlumeTrainer._run_side_batch.
            "pixel_values": [x.get("pixel_values", None) for x in side_instances],
            "image_grid_thw": [x.get("image_grid_thw", None) for x in side_instances],
            "pixel_values_videos": [x.get("pixel_values_videos", None) for x in side_instances],
            "video_grid_thw": [x.get("video_grid_thw", None) for x in side_instances],
        }
        return batch

    def __call__(self, instances: Sequence[Tuple[dict, dict, bool]]) -> Dict[str, object]:
        qry_instances = [instance[0] for instance in instances]
        pos_instances = [instance[1] for instance in instances]
        real_pairs = torch.tensor([bool(instance[2]) for instance in instances], dtype=torch.bool)

        return {
            "qry": self._collate_side(qry_instances),
            "pos": self._collate_side(pos_instances),
            "plume_real_pair": real_pairs,
        }


def make_plume_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    model_args,
    enable_lazy_tokenization: bool = False,
) -> Dict[str, object]:
    train_dataset = LazyPlumeSFTDataset(
        tokenizer=tokenizer,
        data_args=data_args,
        model_args=model_args,
    )
    train_dataset.use_lazy_tokenization = enable_lazy_tokenization

    if enable_lazy_tokenization:
        print(f"[PLUME][DATA] Lazy tokenization ENABLED (multi-worker mode)")
    else:
        print(f"[PLUME][DATA] Lazy tokenization DISABLED (single-worker mode)")

    data_collator = DataCollatorForPlumeDataset(
        tokenizer=tokenizer,
        dataset=train_dataset if enable_lazy_tokenization else None,
    )
    return {
        "train_dataset": train_dataset,
        "eval_dataset": None,
        "data_collator": data_collator,
    }

