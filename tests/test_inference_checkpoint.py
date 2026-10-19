"""Interrupted prompting must resume only matching runs and preserve every document."""
import json
import re

import pytest

from germeval_llm.inference import run_task_c_inference_with_generator, run_task_d_inference_with_generator
from germeval_llm.schemas import CanonicalDocument
from germeval_llm.convert import derive_task_c_example, derive_task_d_example


def docs():
    return [CanonicalDocument(doc_id=str(i), source_split='dev', source_ordinal=i,
                              text=f'TARGET_{i} ' + 'x' * length, relevance=False,
                              document_sentiment='neutral', opinions=[])
            for i, length in enumerate([80, 10, 40, 5, 20])]


def example(task, i):
    document = CanonicalDocument(doc_id=f'train-{i}', source_split='train', source_ordinal=0,
                                 text=f'DEMO_{i}', relevance=False,
                                 document_sentiment='neutral', opinions=[])
    return (derive_task_c_example if task == 'C' else derive_task_d_example)(document)


@pytest.mark.parametrize('task', ['C', 'D'])
@pytest.mark.parametrize('shots', [0, 2])
def test_checkpoint_resume_after_interruption_with_length_grouping(tmp_path, task, shots):
    examples = [example(task, i) for i in range(shots)]
    run = run_task_c_inference_with_generator if task == 'C' else run_task_d_inference_with_generator
    calls = []
    interrupt = [True]

    def batch(prompts):
        if interrupt[0] and calls:
            raise RuntimeError('simulated interruption')
        results = []
        for prompt in prompts:
            index = int(re.search(r'TARGET_(\d+)', prompt).group(1))
            assert all(f'DEMO_{i}' in prompt for i in range(shots))
            calls.append(index)
            raw = '{"labels": []}' if task == 'C' else '{"targets": []}'
            results.append((raw, 10, 5, 2.0))
        return results

    batch.validate_prompts = lambda prompts: [len(prompt) for prompt in prompts]
    options = dict(canonical_docs=docs(), run_id='resume-test', split='dev', generate_fn=None,
                   few_shot_examples=examples, generate_batch_fn=batch, batch_size=2,
                   checkpoint_path=tmp_path / 'state.json', checkpoint_settings={'model_revision': 'pinned'},
                   group_by_length=True)
    with pytest.raises(RuntimeError, match='interruption'):
        run(**options)
    assert calls == [3, 1]
    saved = json.loads(options['checkpoint_path'].read_text())
    assert [row['doc_id'] for row in saved['predictions']] == ['1', '3']
    interrupt[0] = False
    events = []
    predictions = run(**options, progress_callback=lambda *event: events.append(event))
    assert calls == [3, 1, 4, 2, 0]
    assert [record.doc_id for record in predictions] == ['0', '1', '2', '3', '4']
    assert events[0][:2] == (2, 5)
    # A complete checkpoint needs neither generation nor further prompt validation.
    assert run(**options) == predictions
    assert len(calls) == 5
    with pytest.raises(ValueError, match='settings/input mismatch'):
        run(**dict(options, checkpoint_settings={'model_revision': 'different'}))
    changed_docs = docs()
    changed_docs[0] = changed_docs[0].model_copy(update={'text': 'changed input'})
    with pytest.raises(ValueError, match='settings/input mismatch'):
        run(**dict(options, canonical_docs=changed_docs))
    with pytest.raises(ValueError, match='settings/input mismatch'):
        run(**dict(options, few_shot_examples=[] if examples else [example(task, 0)]))
    saved['predictions'][0]['raw_output'] = 'corrupt'
    options['checkpoint_path'].write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='checksum mismatch'):
        run(**options)


def test_checkpoint_atomic_save_keeps_old_file_if_replace_fails(tmp_path, monkeypatch):
    import germeval_llm.inference_checkpoint as checkpoint
    from germeval_llm.schemas import PredictionRecord

    store = checkpoint.InferenceCheckpoint(tmp_path / 'state.json', {'model': 'fixed'})
    record = PredictionRecord(run_id='r', task='C', split='dev', doc_id='0', source_ordinal=0,
                              raw_output='{"labels": []}', parse_status='VALID_EXACT')
    store.save({0: record})
    original = store.path.read_bytes()

    def fail(*args):
        raise OSError('simulated write interruption')

    monkeypatch.setattr(checkpoint.os, 'replace', fail)
    with pytest.raises(OSError, match='interruption'):
        store.save({})
    assert store.path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [store.path]


def test_adapter_content_fingerprint_changes_when_weights_change(tmp_path):
    from germeval_llm.inference_checkpoint import adapter_fingerprint

    weights = tmp_path / 'adapter_model.safetensors'
    weights.write_bytes(b'old weights')
    before = adapter_fingerprint(tmp_path)
    assert adapter_fingerprint(tmp_path) == before
    weights.write_bytes(b'new weights')
    assert adapter_fingerprint(tmp_path) != before


def test_model_checkpoint_settings_bind_resolved_revision_precision_and_adapter(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from germeval_llm import inference
    from germeval_llm.inference_checkpoint import model_inference_settings

    monkeypatch.setattr(inference, 'detect_compute_device', lambda: {'device': 'cuda', 'gpu_name': 'test GPU'})
    config = SimpleNamespace(_name_or_path='test-model', _commit_hash='resolved-sha',
                             to_dict=lambda: {'max_position_embeddings': 8192})
    model = SimpleNamespace(config=config, is_loaded_in_4bit=True,
                            generation_config=SimpleNamespace(to_dict=lambda: {'num_beams': 1}),
                            parameters=lambda: iter([SimpleNamespace(dtype='bfloat16')]))
    tokenizer = SimpleNamespace(init_kwargs={'_commit_hash': 'tokenizer-sha'})
    weights = tmp_path / 'adapter.safetensors'
    weights.write_bytes(b'first')
    settings = model_inference_settings(model, tokenizer, 128, 'main', tmp_path)
    assert settings['model_revision'] == 'resolved-sha'
    assert settings['tokenizer_revision'] == 'tokenizer-sha'
    assert settings['loaded_in_4bit'] is True
    assert settings['max_new_tokens'] == 128 and settings['do_sample'] is False
    weights.write_bytes(b'changed')
    assert model_inference_settings(model, tokenizer, 128, 'main', tmp_path)['adapter_sha256'] != settings['adapter_sha256']
