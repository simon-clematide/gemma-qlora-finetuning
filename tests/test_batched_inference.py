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

    def single(prompt):
        raw = 'invalid JSON' if 'Text 1' in prompt else '{"labels": []}'
        return raw, 10, 5, 2.0

    def batch(prompts):
        calls.append(len(prompts))
        return [single(prompt) for prompt in prompts]

    sequential = run_task_c_inference_with_generator(docs, 'test', 'test_syn', single)
    batched = run_task_c_inference_with_generator(docs, 'test', 'test_syn', single,
                                                 generate_batch_fn=batch, batch_size=2)
    assert calls == [2, 2, 1]
    assert [p.doc_id for p in batched] == ['0', '1', '2', '3', '4']
    assert batched == sequential
    assert batched[1].raw_output == 'invalid JSON'
    assert batched[1].parse_status == 'INVALID_JSON'
    with pytest.raises(ValueError, match='different number'):
        run_task_c_inference_with_generator(docs, 'test', 'test_syn', single,
                                            generate_batch_fn=lambda prompts: [], batch_size=2)
    with pytest.raises(ValueError, match='positive'):
        run_task_c_inference_with_generator(docs, 'test', 'test_syn', single, batch_size=0)
