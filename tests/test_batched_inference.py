"""Batched generation must preserve document alignment and unpadded token counts."""
from types import SimpleNamespace

import pytest

from germeval_llm.inference import build_hf_batch_generator_fn, run_task_c_inference_with_generator
from germeval_llm.schemas import CanonicalDocument


def test_left_padding_and_stop_token_counts():
    torch = pytest.importorskip('torch')

    class Inputs(dict):
        def to(self, device):
            return self

    class Tokenizer:
        padding_side = 'right'
        eos_token_id = 2
        unk_token_id = 99
        pad_token_id = 0

        def convert_tokens_to_ids(self, token):
            return 3

        def __call__(self, prompts, **kwargs):
            assert self.padding_side == 'left'
            assert kwargs == dict(padding=True, return_tensors='pt', add_special_tokens=False)
            return Inputs(input_ids=torch.tensor([[0, 0, 1, 5], [1, 6, 7, 8]]),
                          attention_mask=torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]]))

        def batch_decode(self, rows, **kwargs):
            assert rows == [[10, 3], [11, 12, 13, 2]]
            return [' {"labels": []} ', '{"labels": ["Zugfahrt:negative"]}']

    class Model:
        def parameters(self):
            yield SimpleNamespace(device='cpu')

        def generate(self, **kwargs):
            assert kwargs['do_sample'] is False
            assert kwargs['eos_token_id'] == [2, 3]
            assert kwargs['max_new_tokens'] == 4
            return torch.tensor([[0, 0, 1, 5, 10, 3, 0, 0],
                                 [1, 6, 7, 8, 11, 12, 13, 2]])

    tokenizer = Tokenizer()
    generate = build_hf_batch_generator_fn(Model(), tokenizer, max_new_tokens=4)
    result = generate(['short', 'long'])
    assert [(row[1], row[2]) for row in result] == [(2, 2), (4, 4)]
    assert result[0][0] == '{"labels": []}'
    assert result[0][3] == result[1][3] and result[0][3] > 0
    assert tokenizer.padding_side == 'right'
    assert generate([]) == []


def test_batched_predictions_preserve_order_and_final_partial_batch():
    docs = [CanonicalDocument(doc_id=str(i), source_split='test_syn', source_ordinal=i,
                              text=f'Text {i}', relevance=False, document_sentiment='neutral', opinions=[])
            for i in range(5)]
    calls = []
    progress = []

    def single(prompt):
        raw = 'invalid JSON' if 'Text 1' in prompt else '{"labels": []}'
        return raw, 10, 5, 2.0

    def batch(prompts):
        calls.append(len(prompts))
        return [single(prompt) for prompt in prompts]

    sequential = run_task_c_inference_with_generator(docs, 'test', 'test_syn', single)
    batched = run_task_c_inference_with_generator(docs, 'test', 'test_syn', single,
                                                 generate_batch_fn=batch, batch_size=2,
                                                 progress_callback=lambda *event: progress.append(event))
    assert calls == [2, 2, 1]
    assert [event[0] for event in progress] == [0, 2, 4, 5]
    assert all(event[1] == 5 for event in progress)
    assert [event[2] for event in progress] == sorted(event[2] for event in progress)
    assert [p.doc_id for p in batched] == ['0', '1', '2', '3', '4']
    assert batched == sequential
    assert batched[1].raw_output == 'invalid JSON'
    assert batched[1].parse_status == 'INVALID_JSON'
    with pytest.raises(ValueError, match='different number'):
        run_task_c_inference_with_generator(docs, 'test', 'test_syn', single,
                                            generate_batch_fn=lambda prompts: [], batch_size=2)
    with pytest.raises(ValueError, match='positive'):
        run_task_c_inference_with_generator(docs, 'test', 'test_syn', single, batch_size=0)


@pytest.mark.parametrize('vram,expected', [(14.56, 2), (16, 2), (22.49, 8), (None, 2)])
def test_inference_batch_default_matches_gpu_memory(vram, expected):
    from germeval_llm.inference import resolve_inference_batch_size

    assert resolve_inference_batch_size({'cuda_available': True, 'gpu_vram_gb': vram}) == expected
    assert resolve_inference_batch_size({'cuda_available': True}, requested=4) == 4
    assert resolve_inference_batch_size({'cuda_available': False}) == 1
    with pytest.raises(ValueError, match='positive integer'):
        resolve_inference_batch_size({}, requested=0)


def test_cuda_oom_retry_releases_failed_tensors_and_keeps_every_document(monkeypatch):
    import weakref

    torch = pytest.importorskip('torch')
    failed_tensors = []
    cleanup = []

    def empty_cache():
        assert all(reference() is None for reference in failed_tensors)
        cleanup.append(True)

    monkeypatch.setattr(torch.cuda, 'empty_cache', empty_cache)

    class Inputs(dict):
        def to(self, device):
            return self

    class Tokenizer:
        padding_side = 'right'
        eos_token_id = 2
        unk_token_id = 99
        pad_token_id = 0

        def convert_tokens_to_ids(self, token):
            return 3

        def __call__(self, prompts, **kwargs):
            ids = torch.tensor([[1, int(prompt) + 10] for prompt in prompts])
            return Inputs(input_ids=ids, attention_mask=torch.ones_like(ids))

        def batch_decode(self, rows, **kwargs):
            return [str(row[0] - 10) for row in rows]

    class Model:
        limit = 2
        def __init__(self):
            self.calls = []
            self.transient_failures = 0

        def parameters(self):
            yield SimpleNamespace(device='cpu')

        def generate(self, **kwargs):
            ids = kwargs['input_ids']
            self.calls.append(len(ids))
            if self.transient_failures:
                self.transient_failures -= 1
                raise torch.cuda.OutOfMemoryError('simulated fragmentation')
            if len(ids) > self.limit:
                temporary = torch.zeros(2)
                failed_tensors.append(weakref.ref(temporary))
                raise torch.cuda.OutOfMemoryError('simulated CUDA memory exhaustion')
            return torch.cat([ids, ids[:, 1:], torch.full((len(ids), 1), 2)], dim=1)

    model = Model()
    messages = []
    generate = build_hf_batch_generator_fn(model, Tokenizer(), status_callback=messages.append)
    results = generate([str(i) for i in range(8)])
    assert [result[0] for result in results] == [str(i) for i in range(8)]
    assert model.calls == [8, 8, 4, 4, 2, 2, 2, 2]
    assert len(cleanup) == 4 and len(messages) == 4
    assert generate.batch_statistics['oom_retries'] == 4
    assert generate.batch_statistics['cache_clear_retries'] == 2
    assert generate.batch_statistics['batch_reductions'] == 2
    assert generate.batch_statistics['successful_batch_sizes'] == {'2': 4}
    assert generate.batch_statistics['completed_documents'] == 8
    assert [result[0] for result in generate(['8', '9', '10'])] == ['8', '9', '10']
    assert model.calls[-2:] == [2, 1]  # Reduced capacity persists across outer batches.
    assert generate.batch_statistics['completed_documents'] == 11
    model.limit = 0
    with pytest.raises(torch.cuda.OutOfMemoryError, match='cannot fit even one document'):
        generate(['11'])
    assert len(cleanup) == 6  # Single-document recovery retries once, then stops.

    transient = Model()
    transient.limit = 8
    transient.transient_failures = 1
    transient_generate = build_hf_batch_generator_fn(transient, Tokenizer())
    recovered = transient_generate(['0', '1', '2', '3'])
    assert [row[0] for row in recovered] == ['0', '1', '2', '3']
    assert transient.calls == [4, 4]
    assert transient_generate.batch_statistics['cache_clear_retries'] == 1
    assert transient_generate.batch_statistics['batch_reductions'] == 0

    def invalid(**kwargs):
        raise RuntimeError('unrelated model failure')

    model.generate = invalid
    with pytest.raises(RuntimeError, match='unrelated model failure'):
        generate(['12'])


def test_context_guard_reserves_output_tokens_without_truncating():
    pytest.importorskip('torch')
    model = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device='cpu')]),
                            config=SimpleNamespace(max_position_embeddings=10))
    tokenizer = SimpleNamespace(eos_token_id=2, unk_token_id=99, pad_token_id=0,
                                convert_tokens_to_ids=lambda token: 3,
                                encode=lambda text, **kwargs: list(range(len(text))))
    generate = build_hf_batch_generator_fn(model, tokenizer, max_new_tokens=4)
    assert generate.validate_prompts(['123456']) == [6]
    with pytest.raises(ValueError, match='no input or demonstrations were truncated'):
        generate.validate_prompts(['1234567'])
