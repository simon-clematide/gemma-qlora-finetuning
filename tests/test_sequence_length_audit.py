"""Preflight must count end markers and BOS before deciding whether data fits."""
from types import SimpleNamespace

import pytest

from germeval_llm.training import audit_training_sequence_lengths


def test_sequence_audit_includes_special_tokens_and_records_selected_splits():
    seen = []

    def encode(text):
        seen.append(text)
        return [0] + list(text)  # A tokenizer-added BOS plus the complete text.

    tokenizer = SimpleNamespace(eos_token='!', encode=encode)
    rows = [{'doc_id': 'a', 'prompt': 'p', 'completion': 'c'},
            {'doc_id': 'b', 'prompt': 'pp', 'completion': 'c!'}]
    messages = []
    result = audit_training_sequence_lengths({'train': rows, 'dev': rows[:1]}, tokenizer, 5, messages.append)
    assert seen == ['pc!', 'ppc!', 'pc!']  # Existing EOS is not duplicated.
    assert result['chosen_limit'] == 5
    assert result['measured_max'] == 5
    assert result['splits']['train'] == dict(num_examples=2, median=4, p95=5, p99=5,
                                           max=5, exceeding_limit=0)
    assert result['splits']['dev']['max'] == 4
    assert any('Sequence lengths (train)' in message for message in messages)
    assert rows[0]['completion'] == 'c'  # Input records remain intact.


def test_sequence_audit_rejects_over_limit_train_or_dev_before_loading_weights():
    tokenizer = SimpleNamespace(eos_token='!', encode=lambda text: [0] + list(text))
    short = [{'doc_id': 'short', 'prompt': 'p', 'completion': 'c'}]
    long = [{'doc_id': 'long', 'prompt': 'ppp', 'completion': 'c'}]
    for datasets in [{'train': long, 'dev': short}, {'train': short, 'dev': long}]:
        with pytest.raises(ValueError, match='measured maximum=6') as failure:
            audit_training_sequence_lengths(datasets, tokenizer, 5)
        assert 'suggested 128' in str(failure.value)
        assert 'No model weights have been loaded' in str(failure.value)
    with pytest.raises(ValueError, match='no train examples'):
        audit_training_sequence_lengths({'train': []}, tokenizer, 5)
