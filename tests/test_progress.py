"""Progress must show real completed work while limiting notebook output."""
import sys
from types import ModuleType, SimpleNamespace

from germeval_llm import progress


def test_progress_reports_start_first_batch_periodic_and_completion():
    messages = []
    reporter = progress.ProgressReporter('Baseline documents', messages.append)
    for completed, elapsed in [(0, 0), (8, 1), (16, 2), (24, 32), (32, 33)]:
        reporter(completed, 32, elapsed)
    assert len(messages) == 4
    assert '0/32' in messages[0] and 'ETA estimating' in messages[0]
    assert '8/32' in messages[1]
    assert '24/32' in messages[2]
    assert '32/32' in messages[3] and 'ETA 0.0 min' in messages[3]


def test_training_callback_reports_steps_validation_and_saving(monkeypatch):
    transformers = ModuleType('transformers')
    transformers.TrainerCallback = object
    monkeypatch.setitem(sys.modules, 'transformers', transformers)
    clock = [100.0]
    monkeypatch.setattr(progress.time, 'perf_counter', lambda: clock[0])
    messages = []
    callback = progress.training_progress_callback(messages.append)
    state = SimpleNamespace(global_step=0, max_steps=100)
    callback.on_train_begin(None, state, None)
    state.global_step = 1
    clock[0] = 101.0
    callback.on_step_end(None, state, SimpleNamespace(should_evaluate=True))
    callback.on_evaluate(None, state, None, metrics={'eval_loss': 0.25})
    callback.on_save(None, state, None)
    state.global_step = 100
    clock[0] = 120.0
    callback.on_step_end(None, state, None)
    assert '0/100' in messages[0]
    assert '1/100' in messages[1]
    assert 'Validation starting at step 1' in messages[2]
    assert 'loss=0.25' in messages[3]
    assert 'Checkpoint saved at step 1' in messages[4]
    assert '100/100' in messages[5]


def test_resume_eta_uses_only_work_done_in_current_session():
    messages = []
    reporter = progress.ProgressReporter('Documents', messages.append)
    reporter(80, 100, 0)
    reporter(85, 100, 60)
    assert 'ETA estimating' in messages[0]
    assert 'ETA 3.0 min' in messages[1]
