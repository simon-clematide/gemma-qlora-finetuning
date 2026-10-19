"""Small, flushed status reports for long-running notebook stages."""
from __future__ import annotations

from datetime import datetime
from typing import Callable
import time


def status(message: str) -> None:
    print(f'[{datetime.now().astimezone():%H:%M:%S %Z}] {message}', flush=True)


class ProgressReporter:
    """Report the first batch, then at most every 30 seconds, and completion."""

    def __init__(self, label: str, report: Callable[[str], None] = status):
        self.label = label
        self.report = report
        self.last_elapsed = -30.0
        self.last_completed = None

    def __call__(self, completed: int, total: int, elapsed: float) -> None:
        if completed not in (0, total) and self.last_completed not in (None, 0) and elapsed - self.last_elapsed < 30:
            return
        self.last_elapsed = elapsed
        self.last_completed = completed
        rate = completed / elapsed if elapsed > 0 else 0
        eta = f'{(total - completed) / rate / 60:.1f} min' if rate else 'estimating'
        self.report(f'{self.label}: {completed:,}/{total:,}; elapsed {elapsed / 60:.1f} min; ETA {eta}')


def training_progress_callback(report: Callable[[str], None]):
    """Build a Trainer callback lazily so ordinary analysis needs no ML imports."""
    from transformers import TrainerCallback

    class Callback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            self.started = time.perf_counter()
            self.progress = ProgressReporter('Training optimizer steps', report)
            self.progress(0, state.max_steps, 0)

        def on_step_end(self, args, state, control, **kwargs):
            self.progress(state.global_step, state.max_steps, time.perf_counter() - self.started)
            if getattr(control, "should_evaluate", False):
                report(f"Validation starting at step {state.global_step:,}...")

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            report(f'Validation complete at step {state.global_step:,}; loss={(metrics or {}).get("eval_loss", "unavailable")}')

        def on_save(self, args, state, control, **kwargs):
            report(f'Checkpoint saved at step {state.global_step:,}')

    return Callback()
