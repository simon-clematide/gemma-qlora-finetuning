"""Atomic, settings-bound checkpoints for measured prompting runs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

from germeval_llm.schemas import CanonicalDocument, PredictionRecord


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def adapter_fingerprint(path: str | Path | None) -> str | None:
    if path is None:
        return None
    root = Path(path)
    files = sorted(p for p in root.rglob('*') if p.is_file())
    if not files:
        raise ValueError(f'No adapter files found: {root}')
    hashes = {}
    for file in files:
        digest = hashlib.sha256()
        with file.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        hashes[str(file.relative_to(root))] = digest.hexdigest()
    return fingerprint(hashes)


class InferenceCheckpoint:
    def __init__(self, path: str | Path, identity: dict[str, Any]):
        self.path = Path(path)
        self.signature = fingerprint(identity)

    def load(self, docs: Sequence[CanonicalDocument], task: str, split: str,
             run_id: str) -> dict[int, PredictionRecord]:
        if not self.path.exists():
            return {}
        payload = json.loads(self.path.read_text(encoding='utf-8'))
        if payload.get('format_version') != 1 or payload.get('signature') != self.signature:
            raise ValueError('Inference checkpoint settings/input mismatch; choose a new checkpoint path.')
        rows = payload['predictions']
        if payload.get('predictions_sha256') != fingerprint(rows):
            raise ValueError('Inference checkpoint prediction checksum mismatch')
        indices = {doc.doc_id: i for i, doc in enumerate(docs)}
        records = {}
        previous = -1
        for row in rows:
            record = PredictionRecord.model_validate(row)
            index = indices.get(record.doc_id)
            if index is None or index <= previous or record.task != task or record.split != split or record.run_id != run_id:
                raise ValueError('Inference checkpoint contains unknown, duplicate, or mismatched prediction IDs')
            if record.source_ordinal != docs[index].source_ordinal:
                raise ValueError('Inference checkpoint source order mismatch')
            records[index] = record
            previous = index
        return records

    def save(self, records: dict[int, PredictionRecord]) -> None:
        rows = [records[index].model_dump(mode='json') for index in sorted(records)]
        payload = dict(format_version=1, signature=self.signature, predictions=rows,
                       predictions_sha256=fingerprint(rows))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=self.path.parent,
                                             prefix=self.path.name + '.', delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(payload, stream, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def model_inference_settings(model: Any, tokenizer: Any, max_new_tokens: int,
                             revision: str, adapter_path: str | Path | None = None) -> dict[str, Any]:
    """Bind resumable results to the loaded model, adapter, precision, and decoding."""
    from germeval_llm.inference import detect_compute_device

    config = model.config
    generation = getattr(model, 'generation_config', None)
    hardware = detect_compute_device()
    return {
        'base_model': getattr(config, '_name_or_path', None),
        'model_revision': getattr(config, '_commit_hash', None) or revision,
        'model_config': config.to_dict(),
        'tokenizer_revision': getattr(tokenizer, 'init_kwargs', {}).get('_commit_hash', revision),
        'tokenizer_class': type(tokenizer).__name__,
        'adapter_sha256': adapter_fingerprint(adapter_path),
        'loaded_in_4bit': bool(getattr(model, 'is_loaded_in_4bit', False)),
        'dtype': str(next(model.parameters()).dtype),
        'device': hardware['device'], 'gpu_name': hardware.get('gpu_name'),
        'torch_version': hardware.get('torch_version'),
        'generation_config': generation.to_dict() if generation is not None else None,
        'max_new_tokens': max_new_tokens, 'do_sample': False, 'add_special_tokens': False,
    }
