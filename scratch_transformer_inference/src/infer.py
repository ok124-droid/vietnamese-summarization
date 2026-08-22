#!/usr/bin/env python3
"""Suy luận cho Scratch Transformer tóm tắt tiếng Việt.

Script này tương thích với kiến trúc và checkpoint của dự án hiện tại:

- Model: ``src.model.transformer.TransformerSeq2Seq``
- Checkpoint đầy đủ: ``{"model": state_dict, "config": ...}``
- Checkpoint rút gọn: ``{"model": state_dict}``
- Manifest chuẩn: ``id``, ``source``, ``reference``, ``split``
- Prediction chuẩn: ``id``, ``system``, ``config_id``, ``prediction``,
  ``status``, ``error``

Ví dụ chạy một GPU::

    python src/infer.py \
      --input data/manifests/test_smoke_10.jsonl \
      --output predictions/smoke_10/scratch_transformer.jsonl \
      --checkpoint checkpoints/transformer_base/best_val_loss.pt \
      --tokenizer data/tokenizer/vietnamese_spm.model \
      --config configs/transformer_summarization.yaml \
      --strategy beam_search --beam-size 4 --length-penalty 1.1 \
      --no-repeat-ngram-size 3 --config-id beam4_lp1.1_nr3

Ví dụ dùng hai GPU::

    torchrun --standalone --nproc_per_node=2 src/infer.py <các tham số như trên>

Chỉ nạp checkpoint do bạn hoặc thành viên tin cậy tạo ra. ``torch.load`` có thể
thực thi dữ liệu pickle trong checkpoint cũ.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml


# Cho phép chạy cả ``python src/infer.py`` và ``python -m src.infer``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.transformer import TransformerSeq2Seq  # noqa: E402


try:
    from tqdm.auto import tqdm
except ImportError:

    def tqdm(iterable: Iterable[Any], *args: Any, **kwargs: Any) -> Iterable[Any]:
        return iterable


PREDICTION_FIELDS = (
    "id",
    "system",
    "config_id",
    "prediction",
    "status",
    "error",
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sinh tóm tắt bằng Scratch Transformer từ manifest JSONL chuẩn."
    )

    files = parser.add_argument_group("Tệp đầu vào và mô hình")
    files.add_argument("--input", required=True, help="Manifest JSONL đầu vào.")
    files.add_argument("--output", required=True, help="Prediction JSONL đầu ra.")
    files.add_argument("--checkpoint", required=True, help="Checkpoint .pt đã huấn luyện.")
    files.add_argument("--tokenizer", required=True, help="SentencePiece .model dùng khi train.")
    files.add_argument(
        "--config",
        default=None,
        help="Config YAML. Nếu bỏ qua, script dùng config được nhúng trong checkpoint.",
    )
    files.add_argument(
        "--decoding-config",
        default=None,
        help="Config JSON decoding riêng; tham số CLI sẽ được ưu tiên hơn file này.",
    )

    schema = parser.add_argument_group("Schema")
    schema.add_argument(
        "--source-field",
        default=None,
        help="Tên trường chứa văn bản nguồn. Mặc định ưu tiên 'source' rồi mới thử schema cũ.",
    )
    schema.add_argument(
        "--system",
        choices=("scratch_transformer",),
        default="scratch_transformer",
        help="Tên hệ thống cố định theo schema prediction của dự án.",
    )
    schema.add_argument(
        "--config-id",
        default=None,
        help="ID cấu hình ghi vào prediction; nếu bỏ qua sẽ được tạo tự động.",
    )

    decoding = parser.add_argument_group("Giải mã")
    decoding.add_argument(
        "--strategy",
        choices=("greedy", "beam_search"),
        default=None,
        help="Mặc định lấy từ config.decoding.decoding_strategy.",
    )
    decoding.add_argument(
        "--beam-size", "--num-beams", dest="beam_size", type=int, default=None
    )
    decoding.add_argument("--length-penalty", type=float, default=None)
    decoding.add_argument("--no-repeat-ngram-size", type=int, default=None)
    decoding.add_argument(
        "--max-new-tokens",
        "--max-length",
        dest="max_new_tokens",
        type=int,
        default=None,
    )
    decoding.add_argument(
        "--min-new-tokens",
        "--min-length",
        dest="min_new_tokens",
        type=int,
        default=None,
    )

    runtime = parser.add_argument_group("Thiết bị và cách chạy")
    runtime.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda hoặc cuda:N. Khi dùng torchrun, mỗi rank tự nhận một GPU.",
    )
    runtime.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Số bài báo mã hóa/suy luận cùng một batch. Mặc định 1 để dễ tái lập.",
    )
    runtime.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16"),
        default=None,
        help="Precision suy luận. Mặc định fp32; fp16/bf16 chỉ bật khi yêu cầu rõ ràng.",
    )
    runtime.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Alias tương thích cũ: --amp tương đương --precision fp16.",
    )
    runtime.add_argument(
        "--compile",
        action="store_true",
        help="Dùng torch.compile nếu có. Mặc định tắt để tránh tốn thời gian khởi động.",
    )
    runtime.add_argument("--seed", type=int, default=2026)
    runtime.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Chỉ chạy N mẫu đầu, hữu ích cho smoke test.",
    )
    runtime.add_argument("--quiet", action="store_true")

    writing = parser.add_argument_group("Ghi và khôi phục kết quả")
    mode = writing.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume",
        action="store_true",
        help="Tiếp tục từ output/part files hiện có và bỏ qua ID đã xử lý.",
    )
    mode.add_argument(
        "--overwrite",
        action="store_true",
        help="Ghi đè output/part files hiện có.",
    )
    writing.add_argument(
        "--retry-errors",
        action="store_true",
        help="Cờ tương thích cũ; --resume hiện luôn chạy lại ID status='error'.",
    )

    return parser.parse_args(argv)


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy {label}: {path}")


def load_jsonl(path: Path, tolerate_trailing_partial: bool = False) -> List[Dict[str, Any]]:
    """Đọc JSONL; part file có thể bỏ qua đúng dòng cuối bị ghi dở."""
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            is_last_partial = (
                tolerate_trailing_partial
                and line_number == len(lines)
                and not line.endswith("\n")
            )
            if is_last_partial:
                break
            raise ValueError(f"JSON không hợp lệ tại {path}:{line_number}: {exc}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"Mỗi dòng phải là JSON object tại {path}:{line_number}")
        records.append(item)
    return records


def write_jsonl_atomic(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(f"{path}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def load_manifest(path: Path, limit: Optional[int]) -> List[Dict[str, Any]]:
    records = load_jsonl(path)
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit phải lớn hơn 0.")
        records = records[:limit]
    if not records:
        raise ValueError(f"Manifest rỗng: {path}")

    seen: set[str] = set()
    normalized: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        if "id" not in record:
            raise ValueError(
                f"Mẫu thứ {index} thiếu 'id'. Hãy tạo manifest chuẩn trước khi suy luận."
            )
        sample_id = str(record["id"])
        if not sample_id:
            raise ValueError(f"Mẫu thứ {index} có 'id' rỗng.")
        if sample_id in seen:
            raise ValueError(f"ID bị trùng trong manifest: {sample_id}")
        seen.add(sample_id)
        copied = dict(record)
        copied["id"] = sample_id
        normalized.append(copied)
    return normalized


def load_checkpoint(path: Path) -> Any:
    # Checkpoint này do chính dự án tạo ra và có thể chứa optimizer/config ngoài tensor.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Tương thích PyTorch cũ chưa có tham số weights_only.
        return torch.load(path, map_location="cpu")


def load_config(config_path: Optional[Path], checkpoint: Any) -> Dict[str, Any]:
    if config_path is not None:
        require_file(config_path, "config")
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    elif isinstance(checkpoint, Mapping) and isinstance(checkpoint.get("config"), Mapping):
        config = dict(checkpoint["config"])
    else:
        raise ValueError(
            "Không có --config và checkpoint cũng không chứa trường 'config'."
        )

    if not isinstance(config, dict):
        raise ValueError("Config phải là một YAML mapping/object.")
    for section in ("model", "tokenizer"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Config thiếu section '{section}'.")
    return config


def load_decoding_config(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    require_file(path, "decoding config")
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("Decoding config phải là một JSON object.")
    allowed = {
        "config_id",
        "strategy",
        "decoding_strategy",
        "num_beams",
        "beam_size",
        "length_penalty",
        "no_repeat_ngram_size",
        "max_new_tokens",
        "max_length",
        "min_new_tokens",
        "min_length",
    }
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(
            f"Decoding config có trường không được hỗ trợ: {', '.join(unknown)}"
        )
    if "config_id" in config and (
        not isinstance(config["config_id"], str) or not config["config_id"].strip()
    ):
        raise ValueError("decoding config.config_id phải là chuỗi không rỗng.")
    return config


def extract_state_dict(checkpoint: Any) -> MutableMapping[str, torch.Tensor]:
    candidate: Any = checkpoint
    if isinstance(checkpoint, Mapping):
        for key in ("model", "model_state_dict", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                candidate = value
                break

    if not isinstance(candidate, Mapping) or not candidate:
        raise ValueError("Không tìm thấy model state_dict trong checkpoint.")
    if not all(isinstance(key, str) for key in candidate):
        raise ValueError("Các khóa trong state_dict phải là chuỗi.")
    if not any(torch.is_tensor(value) for value in candidate.values()):
        raise ValueError("Checkpoint không chứa tensor trọng số.")

    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in candidate.items():
        new_key = key
        # DDP tạo 'module.'; torch.compile có thể tạo '_orig_mod.'.
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "_orig_mod."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        cleaned[new_key] = value
    return cleaned


def build_model(config: Mapping[str, Any]) -> TransformerSeq2Seq:
    model_config = config["model"]
    required = (
        "vocab_size",
        "d_model",
        "num_encoder_layers",
        "num_decoder_layers",
        "num_heads",
        "d_ff",
        "max_source_length",
        "max_target_length",
        "pad_id",
    )
    missing = [key for key in required if key not in model_config]
    if missing:
        raise ValueError(f"Config model thiếu trường: {', '.join(missing)}")

    return TransformerSeq2Seq(
        vocab_size=int(model_config["vocab_size"]),
        d_model=int(model_config["d_model"]),
        num_encoder_layers=int(model_config["num_encoder_layers"]),
        num_decoder_layers=int(model_config["num_decoder_layers"]),
        num_heads=int(model_config["num_heads"]),
        d_ff=int(model_config["d_ff"]),
        max_source_length=int(model_config["max_source_length"]),
        max_target_length=int(model_config["max_target_length"]),
        pad_id=int(model_config["pad_id"]),
        dropout=float(model_config.get("dropout", 0.1)),
        activation=str(model_config.get("activation", "gelu")),
        share_encoder_decoder_embeddings=bool(
            model_config.get("share_encoder_decoder_embeddings", True)
        ),
        tie_embeddings=bool(model_config.get("tie_embeddings", True)),
    )


def load_model(
    checkpoint: Any,
    config: Mapping[str, Any],
    device: torch.device,
) -> TransformerSeq2Seq:
    model = build_model(config)
    state_dict = extract_state_dict(checkpoint)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Không nạp được checkpoint. Nguyên nhân thường là config không đúng với "
            "checkpoint hoặc state_dict thuộc kiến trúc khác.\n"
            f"Chi tiết: {exc}"
        ) from exc
    model.to(device)
    model.eval()
    return model


def load_and_validate_tokenizer(
    tokenizer_path: Path,
    config: Mapping[str, Any],
) -> spm.SentencePieceProcessor:
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    tokenizer_config = config["tokenizer"]
    model_config = config["model"]

    expected_vocab = int(model_config["vocab_size"])
    actual_vocab = int(tokenizer.get_piece_size())
    if actual_vocab != expected_vocab:
        raise ValueError(
            f"Tokenizer có vocab_size={actual_vocab}, nhưng model cần {expected_vocab}. "
            "Bạn có thể đang dùng nhầm vietnamese_spm.model."
        )

    id_getters = {
        "pad_id": tokenizer.pad_id,
        "unk_id": tokenizer.unk_id,
        "bos_id": tokenizer.bos_id,
        "eos_id": tokenizer.eos_id,
    }
    for name, getter in id_getters.items():
        if name not in tokenizer_config:
            raise ValueError(f"Config tokenizer thiếu '{name}'.")
        expected = int(tokenizer_config[name])
        actual = int(getter())
        if actual != expected:
            raise ValueError(
                f"Tokenizer {name}={actual}, nhưng config quy định {expected}. "
                "Không được dùng tokenizer huấn luyện lại với checkpoint cũ."
            )
    if int(model_config["pad_id"]) != int(tokenizer_config["pad_id"]):
        raise ValueError("model.pad_id và tokenizer.pad_id trong config không khớp nhau.")
    return tokenizer


def setup_runtime(device_argument: str) -> Tuple[torch.device, int, int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    requested = device_argument.lower()
    if requested == "auto":
        use_cuda = torch.cuda.is_available()
    elif requested == "cpu":
        use_cuda = False
    elif requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("Đã yêu cầu CUDA nhưng torch.cuda.is_available() = False.")
        use_cuda = True
    else:
        raise ValueError("--device chỉ nhận auto, cpu, cuda hoặc cuda:N.")

    if distributed:
        if use_cuda:
            if local_rank >= torch.cuda.device_count():
                raise RuntimeError(
                    f"LOCAL_RANK={local_rank}, nhưng chỉ thấy {torch.cuda.device_count()} GPU."
                )
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            backend = "nccl"
        else:
            device = torch.device("cpu")
            backend = "gloo"
        dist.init_process_group(backend=backend)
    else:
        if use_cuda:
            device = torch.device("cuda" if requested in ("auto", "cuda") else requested)
            if device.index is not None:
                torch.cuda.set_device(device.index)
        else:
            device = torch.device("cpu")

    return device, rank, local_rank, world_size, distributed


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_decoding(
    config: Mapping[str, Any],
    args: argparse.Namespace,
    override: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    decoding = config.get("decoding", {})
    override = override or {}
    strategy = (
        args.strategy
        or override.get("strategy")
        or override.get("decoding_strategy")
        or decoding.get("decoding_strategy", "beam_search")
    )
    beam_size = (
        args.beam_size
        if args.beam_size is not None
        else int(override.get("num_beams", override.get("beam_size", decoding.get("beam_size", 4))))
    )
    length_penalty = (
        args.length_penalty
        if args.length_penalty is not None
        else float(override.get("length_penalty", decoding.get("length_penalty", 1.0)))
    )
    no_repeat_ngram_size = (
        args.no_repeat_ngram_size
        if args.no_repeat_ngram_size is not None
        else int(
            override.get(
                "no_repeat_ngram_size", decoding.get("no_repeat_ngram_size", 0)
            )
        )
    )
    max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else int(
            override.get(
                "max_new_tokens",
                override.get(
                    "max_length",
                    decoding.get("max_length", config["model"]["max_target_length"]),
                ),
            )
        )
    )
    min_new_tokens = (
        args.min_new_tokens
        if args.min_new_tokens is not None
        else int(
            override.get(
                "min_new_tokens",
                override.get("min_length", decoding.get("min_length", 0)),
            )
        )
    )

    if strategy not in ("greedy", "beam_search"):
        raise ValueError(f"Chiến lược giải mã không hỗ trợ: {strategy}")
    if strategy == "greedy":
        beam_size = 1
    if beam_size < 1:
        raise ValueError("beam_size phải >= 1.")
    if length_penalty < 0:
        raise ValueError("length_penalty phải >= 0.")
    if no_repeat_ngram_size < 0:
        raise ValueError("no_repeat_ngram_size phải >= 0.")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens phải >= 1.")
    if min_new_tokens < 0 or min_new_tokens > max_new_tokens:
        raise ValueError("Cần 0 <= min_new_tokens <= max_new_tokens.")

    decoder_capacity = int(config["model"]["max_target_length"])
    if max_new_tokens > decoder_capacity:
        raise ValueError(
            f"max_new_tokens={max_new_tokens} vượt max_target_length="
            f"{decoder_capacity} của positional encoding."
        )

    return {
        "strategy": strategy,
        "beam_size": beam_size,
        "length_penalty": length_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": min_new_tokens,
    }


def make_config_id(decoding: Mapping[str, Any]) -> str:
    if decoding["strategy"] == "greedy":
        name = "greedy"
    else:
        name = f"beam{decoding['beam_size']}_lp{decoding['length_penalty']:g}"
    ngram = int(decoding["no_repeat_ngram_size"])
    if ngram > 0:
        name += f"_nr{ngram}"
    return name


def extract_source(
    sample: Mapping[str, Any],
    explicit_field: Optional[str],
    config: Mapping[str, Any],
) -> str:
    if explicit_field:
        candidates = [explicit_field]
    else:
        configured = config.get("data", {}).get("source_column")
        candidates = ["source"]
        if configured:
            candidates.append(str(configured))
        candidates.extend(("Contents", "article", "content", "document", "text"))

    checked: set[str] = set()
    for field in candidates:
        if field in checked:
            continue
        checked.add(field)
        value = sample.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(
        f"Không tìm thấy source không rỗng. Các trường đã thử: {sorted(checked)}"
    )


def encode_source(
    text: str,
    tokenizer: spm.SentencePieceProcessor,
    max_source_length: int,
    eos_id: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    source_ids = encode_source_ids(text, tokenizer, max_source_length, eos_id)
    source = torch.tensor([source_ids], dtype=torch.long, device=device)
    source_padding_mask = torch.zeros_like(source, dtype=torch.bool)
    return source, source_padding_mask


def encode_source_ids(
    text: str,
    tokenizer: spm.SentencePieceProcessor,
    max_source_length: int,
    eos_id: int,
) -> List[int]:
    if max_source_length < 1:
        raise ValueError("max_source_length phải >= 1.")
    core_ids = tokenizer.encode(text, out_type=int)
    return core_ids[: max_source_length - 1] + [eos_id]


def pad_source_batch(
    sources: Sequence[Sequence[int]],
    pad_id: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not sources:
        raise ValueError("Không thể tạo batch source rỗng.")
    max_length = max(len(source) for source in sources)
    tokens = torch.full(
        (len(sources), max_length), pad_id, dtype=torch.long, device=device
    )
    padding_mask = torch.ones(
        (len(sources), max_length), dtype=torch.bool, device=device
    )
    for row, source in enumerate(sources):
        length = len(source)
        tokens[row, :length] = torch.tensor(source, dtype=torch.long, device=device)
        padding_mask[row, :length] = False
    return tokens, padding_mask


def banned_tokens_for_ngram(tokens: Sequence[int], ngram_size: int) -> set[int]:
    if ngram_size <= 0:
        return set()
    # Bỏ BOS ở đầu khi xét lặp n-gram.
    sequence = list(tokens[1:])
    if ngram_size == 1:
        return set(sequence)
    if len(sequence) < ngram_size - 1:
        return set()

    prefix = tuple(sequence[-(ngram_size - 1):])
    banned: set[int] = set()
    for start in range(0, len(sequence) - ngram_size + 1):
        ngram = sequence[start : start + ngram_size]
        if tuple(ngram[:-1]) == prefix:
            banned.add(ngram[-1])
    return banned


def constrain_logits(
    logits: torch.Tensor,
    tokens: Sequence[int],
    min_new_tokens: int,
    no_repeat_ngram_size: int,
    bos_id: int,
    eos_id: int,
    pad_id: int,
) -> torch.Tensor:
    constrained = logits.clone()
    # Không sinh BOS hoặc PAD ở giữa câu.
    constrained[bos_id] = float("-inf")
    constrained[pad_id] = float("-inf")
    generated_length = len(tokens) - 1
    if generated_length < min_new_tokens:
        constrained[eos_id] = float("-inf")
    for token_id in banned_tokens_for_ngram(tokens, no_repeat_ngram_size):
        constrained[token_id] = float("-inf")
    return constrained


def penalized_score(tokens: Sequence[int], score: float, length_penalty: float) -> float:
    generated_length = max(1, len(tokens) - 1)
    factor = ((5.0 + generated_length) / 6.0) ** length_penalty
    return score / factor


@torch.inference_mode()
def generate_tokens(
    model: TransformerSeq2Seq,
    source: torch.Tensor,
    source_padding_mask: torch.Tensor,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    decoding: Mapping[str, Any],
) -> List[int]:
    encoder_output = model.encode(source, src_pad_mask=source_padding_mask)
    strategy = str(decoding["strategy"])
    max_new_tokens = int(decoding["max_new_tokens"])
    min_new_tokens = int(decoding["min_new_tokens"])
    no_repeat_ngram_size = int(decoding["no_repeat_ngram_size"])
    beam_size = int(decoding["beam_size"])
    length_penalty = float(decoding["length_penalty"])
    device = source.device

    if strategy == "greedy" or beam_size == 1:
        tokens = [bos_id]
        for _ in range(max_new_tokens):
            target = torch.tensor([tokens], dtype=torch.long, device=device)
            logits = model.decode(
                tgt_tokens=target,
                enc_out=encoder_output,
                src_pad_mask=source_padding_mask,
            )[0, -1]
            logits = constrain_logits(
                logits,
                tokens,
                min_new_tokens,
                no_repeat_ngram_size,
                bos_id,
                eos_id,
                pad_id,
            )
            next_token = int(torch.argmax(logits).item())
            tokens.append(next_token)
            if next_token == eos_id:
                break
        return tokens

    active: List[Tuple[List[int], float]] = [([bos_id], 0.0)]
    finished: List[Tuple[List[int], float]] = []

    for _ in range(max_new_tokens):
        if not active:
            break
        target = torch.tensor([tokens for tokens, _ in active], dtype=torch.long, device=device)
        expanded_encoder = encoder_output.expand(len(active), -1, -1)
        expanded_source_mask = source_padding_mask.expand(len(active), -1)
        logits_batch = model.decode(
            tgt_tokens=target,
            enc_out=expanded_encoder,
            src_pad_mask=expanded_source_mask,
        )[:, -1]

        expanded: List[Tuple[List[int], float]] = []
        top_k = min(beam_size, logits_batch.size(-1))
        for beam_index, (tokens, score) in enumerate(active):
            logits = constrain_logits(
                logits_batch[beam_index],
                tokens,
                min_new_tokens,
                no_repeat_ngram_size,
                bos_id,
                eos_id,
                pad_id,
            )
            log_probs = F.log_softmax(logits, dim=-1)
            values, indices = torch.topk(log_probs, k=top_k)
            for value, token_id in zip(values.tolist(), indices.tolist()):
                candidate_tokens = tokens + [int(token_id)]
                candidate_score = score + float(value)
                candidate = (candidate_tokens, candidate_score)
                if token_id == eos_id:
                    finished.append(candidate)
                else:
                    expanded.append(candidate)

        expanded.sort(
            key=lambda item: penalized_score(item[0], item[1], length_penalty),
            reverse=True,
        )
        active = expanded[:beam_size]

    candidates = finished + active
    if not candidates:
        return [bos_id, eos_id]
    candidates.sort(
        key=lambda item: penalized_score(item[0], item[1], length_penalty),
        reverse=True,
    )
    return candidates[0][0]


@torch.inference_mode()
def generate_batch_tokens(
    model: TransformerSeq2Seq,
    source: torch.Tensor,
    source_padding_mask: torch.Tensor,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    decoding: Mapping[str, Any],
) -> List[List[int]]:
    """Sinh cho nhiều source; beam của mọi mẫu được gom vào cùng decoder batch."""
    if source.dim() != 2 or source_padding_mask.shape != source.shape:
        raise ValueError("source và source_padding_mask phải có shape [batch, length].")
    batch_size = int(source.size(0))
    if batch_size < 1:
        return []

    encoder_output = model.encode(source, src_pad_mask=source_padding_mask)
    strategy = str(decoding["strategy"])
    max_new_tokens = int(decoding["max_new_tokens"])
    min_new_tokens = int(decoding["min_new_tokens"])
    no_repeat_ngram_size = int(decoding["no_repeat_ngram_size"])
    beam_size = int(decoding["beam_size"])
    length_penalty = float(decoding["length_penalty"])
    device = source.device

    if strategy == "greedy" or beam_size == 1:
        sequences: List[List[int]] = [[bos_id] for _ in range(batch_size)]
        done = [False] * batch_size
        for _ in range(max_new_tokens):
            target = torch.tensor(sequences, dtype=torch.long, device=device)
            logits_batch = model.decode(
                tgt_tokens=target,
                enc_out=encoder_output,
                src_pad_mask=source_padding_mask,
            )[:, -1]
            for sample_index in range(batch_size):
                if done[sample_index]:
                    sequences[sample_index].append(eos_id)
                    continue
                logits = constrain_logits(
                    logits_batch[sample_index],
                    sequences[sample_index],
                    min_new_tokens,
                    no_repeat_ngram_size,
                    bos_id,
                    eos_id,
                    pad_id,
                )
                next_token = int(torch.argmax(logits).item())
                sequences[sample_index].append(next_token)
                done[sample_index] = next_token == eos_id
            if all(done):
                break
        return sequences

    active: List[List[Tuple[List[int], float]]] = [
        [([bos_id], 0.0)] for _ in range(batch_size)
    ]
    finished: List[List[Tuple[List[int], float]]] = [
        [] for _ in range(batch_size)
    ]

    for _ in range(max_new_tokens):
        flat_beams: List[Tuple[int, List[int], float]] = []
        for sample_index, beams in enumerate(active):
            for tokens, score in beams:
                flat_beams.append((sample_index, tokens, score))
        if not flat_beams:
            break

        target = torch.tensor(
            [tokens for _, tokens, _ in flat_beams], dtype=torch.long, device=device
        )
        sample_indices = torch.tensor(
            [sample_index for sample_index, _, _ in flat_beams],
            dtype=torch.long,
            device=device,
        )
        expanded_encoder = encoder_output.index_select(0, sample_indices)
        expanded_source_mask = source_padding_mask.index_select(0, sample_indices)
        logits_batch = model.decode(
            tgt_tokens=target,
            enc_out=expanded_encoder,
            src_pad_mask=expanded_source_mask,
        )[:, -1]

        expanded: List[List[Tuple[List[int], float]]] = [
            [] for _ in range(batch_size)
        ]
        top_k = min(beam_size, logits_batch.size(-1))
        for beam_index, (sample_index, tokens, score) in enumerate(flat_beams):
            logits = constrain_logits(
                logits_batch[beam_index],
                tokens,
                min_new_tokens,
                no_repeat_ngram_size,
                bos_id,
                eos_id,
                pad_id,
            )
            log_probs = F.log_softmax(logits, dim=-1)
            values, indices = torch.topk(log_probs, k=top_k)
            for value, token_id in zip(values.tolist(), indices.tolist()):
                candidate = (tokens + [int(token_id)], score + float(value))
                if token_id == eos_id:
                    finished[sample_index].append(candidate)
                else:
                    expanded[sample_index].append(candidate)

        for sample_index in range(batch_size):
            expanded[sample_index].sort(
                key=lambda item: penalized_score(
                    item[0], item[1], length_penalty
                ),
                reverse=True,
            )
            active[sample_index] = expanded[sample_index][:beam_size]

    outputs: List[List[int]] = []
    for sample_index in range(batch_size):
        candidates = finished[sample_index] + active[sample_index]
        if not candidates:
            outputs.append([bos_id, eos_id])
            continue
        candidates.sort(
            key=lambda item: penalized_score(item[0], item[1], length_penalty),
            reverse=True,
        )
        outputs.append(candidates[0][0])
    return outputs


def decode_tokens(
    tokens: Sequence[int],
    tokenizer: spm.SentencePieceProcessor,
    bos_id: int,
    eos_id: int,
    pad_id: int,
) -> str:
    clean: List[int] = []
    for token_id in tokens:
        if token_id == eos_id:
            break
        if token_id not in (bos_id, pad_id):
            clean.append(int(token_id))
    return tokenizer.decode(clean).strip()


def resolve_precision(args: argparse.Namespace, device: torch.device) -> str:
    if args.precision is not None and args.amp is not None:
        raise ValueError("Chỉ dùng một trong --precision hoặc --amp/--no-amp.")
    if args.precision is not None:
        precision = str(args.precision)
    elif args.amp is True:
        precision = "fp16"
    else:
        precision = "fp32"
    if precision != "fp32" and device.type != "cuda":
        raise ValueError(f"precision={precision} chỉ được hỗ trợ trên CUDA.")
    if (
        precision == "bf16"
        and hasattr(torch.cuda, "is_bf16_supported")
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("GPU hiện tại không hỗ trợ BF16.")
    return precision


def autocast_context(device: torch.device, precision: str) -> Any:
    if precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    if hasattr(torch, "autocast"):
        return torch.autocast(device_type=device.type, dtype=dtype)
    return torch.cuda.amp.autocast(dtype=dtype)


def part_path(output_path: Path, rank: int) -> Path:
    return Path(f"{output_path}.rank{rank}.part")


def all_part_paths(output_path: Path) -> List[Path]:
    return sorted(output_path.parent.glob(f"{output_path.name}.rank*.part"))


def result_map_from_files(paths: Sequence[Path]) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        if not path.is_file():
            continue
        for item in load_jsonl(path, tolerate_trailing_partial=path.suffix == ".part"):
            if "id" in item:
                results[str(item["id"])] = item
    return results


def normalize_resume_files(
    manifest: Sequence[Mapping[str, Any]],
    output_path: Path,
    system: str,
    config_id: str,
    rank: int,
    distributed: bool,
) -> Tuple[Dict[str, Dict[str, Any]], int]:
    """Giữ record thành công, loại record lỗi để --resume tự chạy lại chúng."""
    paths = [output_path] + all_part_paths(output_path)
    existing = result_map_from_files(paths)
    if existing:
        validate_resume_results(existing, system, config_id)
    manifest_ids = {str(sample["id"]) for sample in manifest}
    extra_ids = sorted(set(existing) - manifest_ids)
    if extra_ids:
        raise ValueError(f"Không thể --resume vì có ID ngoài manifest: {extra_ids[:3]}")

    successful: Dict[str, Dict[str, Any]] = {}
    prior_error_count = 0
    for sample_id, item in existing.items():
        status = item.get("status")
        if status == "ok":
            if not str(item.get("prediction") or "").strip():
                raise ValueError(
                    f"Không thể --resume: ID {sample_id} status='ok' nhưng rỗng."
                )
            successful[sample_id] = {
                field: item.get(field) for field in PREDICTION_FIELDS
            }
        elif status == "error":
            prior_error_count += 1
        else:
            raise ValueError(
                f"Không thể --resume: ID {sample_id} có status không hợp lệ."
            )

    if rank == 0:
        ordered_success = [
            successful[str(sample["id"])]
            for sample in manifest
            if str(sample["id"]) in successful
        ]
        if existing or output_path.exists() or all_part_paths(output_path):
            write_jsonl_atomic(ordered_success, output_path)
        for path in all_part_paths(output_path):
            path.unlink()

    if distributed:
        dist.barrier()
    successful_results = (
        result_map_from_files([output_path]) if output_path.is_file() else {}
    )
    return successful_results, prior_error_count


def batched(
    items: Sequence[Tuple[int, Dict[str, Any]]], batch_size: int
) -> Iterable[Sequence[Tuple[int, Dict[str, Any]]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def make_prediction_record(
    sample_id: str,
    system: str,
    config_id: str,
    index: int,
    prediction: Optional[str] = None,
    error: Optional[BaseException] = None,
) -> Dict[str, Any]:
    if error is None:
        return {
            "id": sample_id,
            "system": system,
            "config_id": config_id,
            "prediction": prediction or "",
            "status": "ok",
            "error": None,
            "_index": index,
        }
    return {
        "id": sample_id,
        "system": system,
        "config_id": config_id,
        "prediction": "",
        "status": "error",
        "error": f"{type(error).__name__}: {error}",
        "_index": index,
    }


def clear_cuda_after_oom(error: BaseException, device: torch.device) -> None:
    if device.type != "cuda":
        return
    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    is_oom = (
        (oom_type is not None and isinstance(error, oom_type))
        or "out of memory" in str(error).lower()
    )
    if is_oom:
        torch.cuda.empty_cache()


def validate_resume_results(
    results: Mapping[str, Mapping[str, Any]],
    system: str,
    config_id: str,
) -> None:
    for sample_id, item in results.items():
        if item.get("system") != system or item.get("config_id") != config_id:
            raise ValueError(
                f"Không thể --resume vì prediction '{sample_id}' dùng system/config_id "
                f"khác: {item.get('system')}/{item.get('config_id')}."
            )


def prepare_output(
    output_path: Path,
    overwrite: bool,
    resume: bool,
    rank: int,
    distributed: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_parts = all_part_paths(output_path)
    exists = output_path.exists() or bool(existing_parts)
    # Mọi rank cùng phát hiện xung đột để không có rank nào bị kẹt ở barrier.
    if exists and not overwrite and not resume:
        raise FileExistsError(
            f"Output đã tồn tại: {output_path}. Dùng --resume hoặc --overwrite."
        )
    if overwrite:
        if rank == 0:
            if output_path.exists():
                output_path.unlink()
            for path in existing_parts:
                path.unlink()
    if distributed:
        dist.barrier()


def merge_results(
    manifest: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> Tuple[int, int]:
    id_to_index = {str(item["id"]): index for index, item in enumerate(manifest)}
    sources = [output_path] + all_part_paths(output_path)
    merged = result_map_from_files(sources)

    unknown_ids = sorted(set(merged) - set(id_to_index))
    if unknown_ids:
        raise ValueError(
            f"Output chứa {len(unknown_ids)} ID không thuộc manifest, ví dụ: {unknown_ids[:3]}"
        )
    missing_ids = [sample_id for sample_id in id_to_index if sample_id not in merged]
    if missing_ids:
        raise RuntimeError(
            f"Thiếu {len(missing_ids)} prediction sau khi gộp, ví dụ: {missing_ids[:3]}"
        )

    final_records: List[Dict[str, Any]] = []
    for sample in manifest:
        sample_id = str(sample["id"])
        raw = merged[sample_id]
        record = {field: raw.get(field) for field in PREDICTION_FIELDS}
        if record["status"] not in ("ok", "error"):
            raise ValueError(f"Prediction {sample_id} có status không hợp lệ.")
        if record["status"] == "ok" and not str(record["prediction"] or "").strip():
            raise ValueError(f"Prediction {sample_id} status='ok' nhưng prediction rỗng.")
        final_records.append(record)

    write_jsonl_atomic(final_records, output_path)
    for path in all_part_paths(output_path):
        path.unlink()

    ok_count = sum(item["status"] == "ok" for item in final_records)
    return ok_count, len(final_records) - ok_count


def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    tokenizer_path = Path(args.tokenizer).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve() if args.config else None
    decoding_config_path = (
        Path(args.decoding_config).expanduser().resolve()
        if args.decoding_config
        else None
    )

    if args.retry_errors and not args.resume:
        raise ValueError("--retry-errors chỉ được dùng cùng --resume.")
    if args.batch_size < 1:
        raise ValueError("--batch-size phải >= 1.")
    protected_paths = {input_path, checkpoint_path, tokenizer_path}
    if config_path is not None:
        protected_paths.add(config_path)
    if decoding_config_path is not None:
        protected_paths.add(decoding_config_path)
    if output_path in protected_paths:
        raise ValueError("--output phải khác input, checkpoint, tokenizer và config.")

    require_file(input_path, "manifest")
    require_file(checkpoint_path, "checkpoint")
    require_file(tokenizer_path, "tokenizer")

    device, rank, local_rank, world_size, distributed = setup_runtime(args.device)
    set_seed(args.seed + rank)

    try:
        prepare_output(output_path, args.overwrite, args.resume, rank, distributed)
        manifest = load_manifest(input_path, args.limit)
        checkpoint = load_checkpoint(checkpoint_path)
        config = load_config(config_path, checkpoint)
        decoding_override = load_decoding_config(decoding_config_path)
        decoding = resolve_decoding(config, args, decoding_override)
        config_id = (
            args.config_id
            or decoding_override.get("config_id")
            or make_config_id(decoding)
        )
        if not str(args.system).strip() or not str(config_id).strip():
            raise ValueError("system và config_id không được rỗng.")
        if args.resume:
            existing, prior_error_count = normalize_resume_files(
                manifest,
                output_path,
                args.system,
                str(config_id),
                rank,
                distributed,
            )
        else:
            existing, prior_error_count = {}, 0
        processed_ids = set(existing)

        tokenizer = load_and_validate_tokenizer(tokenizer_path, config)
        model = load_model(checkpoint, config, device)
        del checkpoint
        gc.collect()

        if args.compile:
            if not hasattr(torch, "compile"):
                raise RuntimeError("Phiên bản PyTorch hiện tại không hỗ trợ torch.compile.")
            model = torch.compile(model)  # type: ignore[assignment]

        precision = resolve_precision(args, device)

        indexed_samples = list(enumerate(manifest))
        assigned = [
            (index, sample)
            for index, sample in indexed_samples
            if index % world_size == rank and str(sample["id"]) not in processed_ids
        ]

        if rank == 0 and not args.quiet:
            print(
                f"Thiết bị={device} | processes={world_size} | samples={len(manifest)} | "
                f"batch_size={args.batch_size} | precision={precision} | "
                f"strategy={decoding['strategy']} | config_id={config_id}\n"
                f"Resume: skipped_ok={len(processed_ids)} | retry_errors={prior_error_count}"
            )

        tokenizer_config = config["tokenizer"]
        bos_id = int(tokenizer_config["bos_id"])
        eos_id = int(tokenizer_config["eos_id"])
        pad_id = int(tokenizer_config["pad_id"])
        max_source_length = int(config["model"]["max_source_length"])

        current_part = part_path(output_path, rank)
        iterator: Iterable[Sequence[Tuple[int, Dict[str, Any]]]] = batched(
            assigned, args.batch_size
        )
        if rank == 0 and not args.quiet:
            num_batches = (len(assigned) + args.batch_size - 1) // args.batch_size
            iterator = tqdm(
                iterator,
                total=num_batches,
                desc=f"Inference rank {local_rank}",
                unit="batch",
            )

        with current_part.open("w", encoding="utf-8", buffering=1) as handle:
            for sample_batch in iterator:
                results: Dict[int, Dict[str, Any]] = {}
                valid: List[Tuple[int, str, List[int]]] = []

                for index, sample in sample_batch:
                    sample_id = str(sample["id"])
                    try:
                        source_text = extract_source(sample, args.source_field, config)
                        source_ids = encode_source_ids(
                            source_text, tokenizer, max_source_length, eos_id
                        )
                        valid.append((index, sample_id, source_ids))
                    except Exception as exc:
                        results[index] = make_prediction_record(
                            sample_id,
                            args.system,
                            str(config_id),
                            index,
                            error=exc,
                        )

                if valid:
                    try:
                        source, source_padding_mask = pad_source_batch(
                            [item[2] for item in valid], pad_id, device
                        )
                        with autocast_context(device, precision):
                            generated_batch = generate_batch_tokens(
                                model,
                                source,
                                source_padding_mask,
                                bos_id,
                                eos_id,
                                pad_id,
                                decoding,
                            )
                        for (index, sample_id, _), generated_tokens in zip(
                            valid, generated_batch
                        ):
                            try:
                                prediction = decode_tokens(
                                    generated_tokens,
                                    tokenizer,
                                    bos_id,
                                    eos_id,
                                    pad_id,
                                )
                                if not prediction:
                                    raise RuntimeError("Mô hình sinh prediction rỗng.")
                                results[index] = make_prediction_record(
                                    sample_id,
                                    args.system,
                                    str(config_id),
                                    index,
                                    prediction=prediction,
                                )
                            except Exception as exc:
                                results[index] = make_prediction_record(
                                    sample_id,
                                    args.system,
                                    str(config_id),
                                    index,
                                    error=exc,
                                )
                    except Exception as batch_exc:
                        clear_cuda_after_oom(batch_exc, device)
                        # Batch lỗi không được làm mất toàn bộ mẫu: thử lại từng mẫu.
                        for index, sample_id, source_ids in valid:
                            try:
                                source, source_padding_mask = pad_source_batch(
                                    [source_ids], pad_id, device
                                )
                                with autocast_context(device, precision):
                                    generated_tokens = generate_batch_tokens(
                                        model,
                                        source,
                                        source_padding_mask,
                                        bos_id,
                                        eos_id,
                                        pad_id,
                                        decoding,
                                    )[0]
                                prediction = decode_tokens(
                                    generated_tokens,
                                    tokenizer,
                                    bos_id,
                                    eos_id,
                                    pad_id,
                                )
                                if not prediction:
                                    raise RuntimeError("Mô hình sinh prediction rỗng.")
                                results[index] = make_prediction_record(
                                    sample_id,
                                    args.system,
                                    str(config_id),
                                    index,
                                    prediction=prediction,
                                )
                            except Exception as exc:
                                clear_cuda_after_oom(exc, device)
                                results[index] = make_prediction_record(
                                    sample_id,
                                    args.system,
                                    str(config_id),
                                    index,
                                    error=exc,
                                )

                for index, _ in sample_batch:
                    handle.write(
                        json.dumps(results[index], ensure_ascii=False) + "\n"
                    )
                    handle.flush()

        if distributed:
            dist.barrier()

        if rank == 0:
            ok_count, error_count = merge_results(manifest, output_path)
            if not args.quiet:
                print(
                    f"Hoàn tất: {output_path}\n"
                    f"Số mẫu: {len(manifest)} | ok: {ok_count} | "
                    f"error: {error_count} | skipped_resume: {len(processed_ids)}"
                )
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    configure_stdio()
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
