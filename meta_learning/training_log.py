"""Readable training progress with complete structured events stored separately."""

from contextvars import ContextVar
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import fmean
import sys


_ACTIVE_LOG = ContextVar('training_log', default=None)


def metric(value):
    """Accuracy percent and predictive loss; absent metrics remain explicit."""
    if not value:
        return '--'
    return f"{value['accuracy'] * 100:.2f}%/{value['loss']:.4f}"


def average(values):
    values = list(values)
    if not values:
        return None
    return {key: fmean(value[key] for value in values) for key in ('accuracy', 'loss')}


def fit_metrics(fit):
    return dict(accuracy=fit['last_support_accuracy'], loss=fit['last_support_loss'])


def duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:d}h{minutes:02d}m{seconds:02d}s' if hours else f'{minutes:d}m{seconds:02d}s'


def task_event(step, mouse, fit, **extra):
    return dict(step=step, mouse=mouse, support=fit_metrics(fit),
                support_new=fit.get('support_crop_validation'), query=fit.get('query'),
                query_new=fit.get('query_crop_validation'), **extra)


def refit_event(step, mouse, reports):
    query = reports['query']['subjects'][mouse]
    fit = query['classifier_fit']
    result = dict(step=step, mouse=mouse, support=fit_metrics(fit),
        support_new=fit.get('support_crop_validation'), query=query,
        query_new=fit.get('query_crop_validation'), test=reports['test']['subjects'][mouse])
    if 'valid' in reports:
        result['valid'] = reports['valid']['subjects'][mouse]
    return result


def readable_lines(stage, row, total=None):
    """Pure formatting: no model, RNG, sampling, or checkpoint state changes."""
    prefix = f"[{row['step']:03d}/{total}]" if 'step' in row and total else (
        f"[{row['step']:03d}]" if 'step' in row else '')
    if stage == 'training_started':
        config = row['config']
        return [
            f"Meta training | {config['outer_steps']} outer steps x {config['inner_epochs']} inner epochs | {row['device']}",
            f"Mice: {', '.join(config['targets'])} | support pool/draw: {config['support_pool_crops']}/{config['support_crops']} per event/class | query crops: {config['query_crops']}",
            f"SGD lr={config['inner_lr']:g}, decay={config['inner_weight_decay']:g} | AdamW lr={config['outer_lr']:g}, decay={config['outer_weight_decay']:g} | dropout={config['classifier_dropout']:g} | support weight={config['support_loss_weight']:g}",
            'Values are accuracy / loss. S=support, Q=query, -> new=independent crop draw.',
            'Task rows: final fitted CNN. MEAN: equal mouse mean. Objective: query-event-weighted outer loss.',
            'Gaps are fitted accuracy minus new-crop accuracy, in percentage points (pp).',
            'EVAL: fresh support-only CNN refits; Q uses fixed windows; test never selects checkpoints.',
            f"Support summary: {row['support_scope']}",
            '']
    if stage in {'task_complete', 'refit_complete'}:
        mode = 'EVAL' if stage == 'refit_complete' else 'META'
        line = (f"{prefix} {mode} {row['mouse']:<3} | S {metric(row.get('support'))} -> new {metric(row.get('support_new'))}"
                f" | Q {metric(row.get('query'))} -> new {metric(row.get('query_new'))}")
        if 'valid' in row:
            line += f" | valid {metric(row['valid'])}"
        if 'test' in row:
            line += f" | test {metric(row['test'])}"
        return [line]
    if stage == 'outer_step':
        support_diagnostics = row.get('support_crop_validation', {}).get('subjects', {})
        support = average(dict(accuracy=value['accuracy'] + value['accuracy_gap'],
                               loss=value['loss'] - value['loss_gap']) for value in support_diagnostics.values())
        query = average(row['query'].get('subjects', {}).values())
        new_support = average(support_diagnostics.values())
        new_query = average(row.get('query_crop_validation', {}).get('subjects', {}).values())
        lines = [f"{prefix} MEAN     | S {metric(support)} -> new {metric(new_support)} | Q {metric(query)} -> new {metric(new_query)}"]
        parts = [f"objective={row['outer_loss']:.4f}"]
        if support and new_support:
            parts.append(f"S gap {(support['accuracy'] - new_support['accuracy']) * 100:+.2f}pp")
        if query and new_query:
            parts.append(f"Q gap {(query['accuracy'] - new_query['accuracy']) * 100:+.2f}pp")
        if 'step_seconds' in row:
            parts.append(f"step {duration(row['step_seconds'])}")
        if 'elapsed_seconds' in row:
            parts.append(f"elapsed {duration(row['elapsed_seconds'])}")
        if 'peak_vram_mb' in row:
            parts.append(f"peak {row['peak_vram_mb'] / 1024:.2f} GiB")
        return lines + ['           ' + ' | '.join(parts), '']
    if stage == 'evaluation_started':
        return [f'{prefix} EVAL: fitting fresh CNNs on support, then scoring query/test.']
    if stage == 'evaluation_step':
        roles = ' | '.join(f"{role} {metric(row[role])}" for role in ('query', 'valid', 'test') if role in row)
        return [f'{prefix} EVAL MEAN | {roles}']
    if stage == 'checkpoint_selected':
        return [f"{prefix} SELECTED  | {row['role']} loss={row['loss']:.4f}"]
    if stage == 'training_complete':
        return [f"Training complete: {row['completed_steps']} steps | selected step {row['selected_step']} by {row['role']} loss={row['loss']:.4f}"]
    if stage == 'training_error':
        return [f"ERROR {row['error']}: {row['message']}"]
    if stage == 'convergence_plots':
        return ['Plot: ' + str(path) for path in row['paths']]
    return []


class TrainingLog:
    def __init__(self, run_dir, config, device, *, stream=None):
        self.path = Path(run_dir) / 'train.events.jsonl'
        self.config = config
        self.device = device
        self.stream = sys.stdout if stream is None else stream

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.events = self.path.open('a', encoding='utf-8', buffering=1)
        self.token = _ACTIVE_LOG.set(self)
        self.emit('training_started', dict(config=self.config.to_dict(), device=str(self.device),
            support_scope='Uniform mean over all pool slots at final CNN/BN state; unused slots included.'))
        return self

    def emit(self, stage, row):
        event = dict(stage=stage, timestamp_utc=datetime.now(timezone.utc).isoformat(), **row)
        self.events.write(json.dumps(event, allow_nan=False) + '\n')
        for line in readable_lines(stage, row, self.config.outer_steps):
            print(line, file=self.stream, flush=True)

    def __exit__(self, error_type, error, traceback):
        try:
            if error is not None:
                self.emit('training_error', dict(error=error_type.__name__, message=str(error)))
        finally:
            _ACTIVE_LOG.reset(self.token)
            self.events.close()


def progress_event(stage, row, *, run_dir=None):
    logger = _ACTIVE_LOG.get()
    if logger is not None:
        logger.emit(stage, row)
    else:
        if run_dir is not None:
            path = Path(run_dir) / 'train.events.jsonl'
            with path.open('a', encoding='utf-8') as events:
                events.write(json.dumps(dict(stage=stage, timestamp_utc=datetime.now(timezone.utc).isoformat(), **row)) + '\n')
        for line in readable_lines(stage, row):
            print(line, flush=True)
