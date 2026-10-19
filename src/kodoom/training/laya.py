"""Adapter to an immutable, checksum-verified upstream Laya implementation."""

from __future__ import annotations

import importlib.util
import json
import urllib.request
from pathlib import Path

from kodoom.package_import import file_sha
from kodoom.schema import Record, RecordError
from kodoom.training.data import model_record

MODEL_ID = "convaiinnovations/laya-multilingual"
MODEL_REVISION = "1720e3e3357cfe1e281542e223f8273b0890ca34"
CODE_REVISION = "ffc142012af63d015fa19f05f29f9c5ff329a2cc"
COMMON_SHA = "8396e8c8dbceabfc7fb2abb1e2fe4692751389ba9bd80d592f1ea922fef0765c"
FORWARD_KEYS = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")


def upstream(cache: Path):
    """Download code only, verify before execution; upstream is Apache-2.0."""
    path = cache / "laya-code" / CODE_REVISION / "common.py"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://raw.githubusercontent.com/NandhaKishorM/laya/{CODE_REVISION}/laya/common.py"
        content = urllib.request.urlopen(url, timeout=60).read()
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(content)
        if file_sha(tmp) != COMMON_SHA:
            raise RecordError("upstream Laya source checksum mismatch")
        tmp.replace(path)
    if file_sha(path) != COMMON_SHA:
        raise RecordError("cached Laya source checksum mismatch")
    spec = importlib.util.spec_from_file_location("kodoom_pinned_laya_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def question(r: Record) -> tuple[dict, list[str]]:
    r = model_record(r)
    if r.question_type == "score":
        if [o.id for o in r.options] != [str(i) for i in range(len(r.options))]:
            raise RecordError(f"{r.id}: score options must be ordered zero-based levels")
        crit = [o.text for o in r.options]
        ids = [o.id for o in r.options]
    elif r.question_type == "noul":
        if {o.id for o in r.options} != {"false", "true"}:
            raise RecordError(f"{r.id}: noul requires semantic false/true option IDs")
        crit = {o.id: o.text for o in r.options}
        ids = ["false", "true"]
    else:
        crit = {o.id: o.text for o in r.options}
        ids = [o.id for o in r.options]
    return {"t": r.question_type, "ins": r.question_text, "crit": crit}, ids


def encode(record: Record, tokenizer, common, *, max_len: int, head_max_len: int) -> dict:
    r = model_record(record)
    q, option_ids = question(r)
    ids, markers, stats, truncation = common.build_sequence(
        tokenizer,
        r.state,
        q,
        max_len=max_len,
        head_max_len=head_max_len,
        return_stats=True,
        return_truncation_stats=True,
    )
    if len(markers) != len(option_ids) or stats["options_distinct"] != len(option_ids):
        raise RecordError(f"{r.id}: option markers lost under head/context budget")
    target = [r.gold.get(o, 0.0) for o in option_ids]
    total = sum(target)
    # Source rounding is tolerated by Record; proper CE still needs normalized mass.
    target = [p / total for p in target]
    return {
        "ids": ids,
        "markers": markers,
        "target": target,
        "qtype": common.QTYPES[r.question_type],
        "record_id": r.id,
        "option_ids": option_ids,
        "truncation": truncation,
        "option_stats": stats,
        "language": r.question_lang,
    }


def load(cache: Path, *, tiny: bool = False, artifact: Path | None = None):
    import torch
    from transformers import AutoTokenizer, BertConfig, BertModel

    common = upstream(cache)
    if artifact:
        root = artifact
    else:
        from huggingface_hub import snapshot_download

        patterns = ["tokenizer/*", "rl_agent_config.json", "encoder/config.json"]
        if not tiny:
            patterns += ["model.safetensors"]
        root = Path(
            snapshot_download(
                MODEL_ID, revision=MODEL_REVISION, cache_dir=str(cache), allow_patterns=patterns
            )
        )
    # This pinned tokenizer was saved with a Transformers-era list format;
    # current releases require the same two added tokens as a name-to-token map.
    tok = AutoTokenizer.from_pretrained(
        root / "tokenizer",
        local_files_only=True,
        extra_special_tokens={
            "start_of_turn": "<start_of_turn>",
            "end_of_turn": "<end_of_turn>",
        },
    )
    cfg = json.loads((root / "rl_agent_config.json").read_text("utf-8"))
    if tiny:
        encoder = BertModel(
            BertConfig(
                vocab_size=len(tok),
                hidden_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=128,
                max_position_embeddings=256,
            ),
            add_pooling_layer=False,
        )
        model = common.DecisionModel(encoder, head_layers=1)
        cfg.update(encoder="tiny-bert-cpu-smoke", head_layers=1, max_len=128, head_max_len=80)
    else:
        from safetensors.torch import load_file

        model = common.build_model(cfg, encoder_dir=str(root / "encoder"), pretrained=False)
        model.load_state_dict(load_file(root / "model.safetensors"), strict=True)
        if hasattr(model.encoder.config, "reference_compile"):
            model.encoder.config.reference_compile = False
    if tiny and artifact:
        from safetensors.torch import load_file

        model.load_state_dict(load_file(root / "model.safetensors"), strict=True)
    # act_head has no target in this recipe; prevent unused trainable DDP parameters.
    for p in model.act_head.parameters():
        p.requires_grad_(False)
    model.temperature.copy_(torch.ones_like(model.temperature))
    cfg["temperature"] = [1.0, 1.0, 1.0]
    return model, tok, cfg, common


def collate(items: list[dict], tokenizer, common, device):
    b = common.collate_items([[item] for item in items], tokenizer.pad_token_id)
    return {k: v.to(device) for k, v in b.items() if k in (*FORWARD_KEYS, "target")}


def save_artifact(path: Path, model, tokenizer, cfg: dict, training: dict):
    from safetensors.torch import save_file

    model.encoder.config.save_pretrained(path / "encoder")
    tokenizer.save_pretrained(path / "tokenizer")
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
        str(path / "model.safetensors"),
    )
    (path / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (path / "training.json").write_text(json.dumps(training, indent=2), encoding="utf-8")
