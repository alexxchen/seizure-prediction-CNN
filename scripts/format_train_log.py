"""Render or follow a readable progress log from immutable run metric artifacts."""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meta_learning.training_log import readable_lines, task_event, refit_event


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def render(run_dir, output, *, follow=False):
    run_dir, output = Path(run_dir), Path(output)
    if output.resolve() in {path.resolve() for path in (
            run_dir/'train.log', run_dir/'history.json', run_dir/'status.json',
            run_dir/'selection.json', run_dir/'config.json', run_dir/'run.json', run_dir/'result.json',
            run_dir/'train.events.jsonl')}:
        raise ValueError('Readable output must not overwrite an original run artifact')
    config = read(run_dir/'config.json')
    metadata = read(run_dir/'run.json')
    first_fits = sorted((run_dir/'inner').glob('step-*/task-*.json'))
    support_scope = read(first_fits[0])['last_support_scope'] if first_fits else 'See original per-task classifier_fit records.'
    output.parent.mkdir(parents=True, exist_ok=True)
    tasks_seen, steps_seen, evaluations_seen = set(), set(), set()
    selected_seen = None
    with output.open('w', encoding='utf-8', buffering=1) as stream:
        def emit(stage, row):
            for line in readable_lines(stage, row, config['outer_steps']):
                print(line, file=stream, flush=True)
        emit('training_started', dict(config=config,
            device=metadata['provenance'].get('runtime', {}).get('device', 'saved runtime'),
            support_scope=support_scope))
        print('Live view from saved metrics; original train.log and run sources are preserved.\n', file=stream, flush=True)
        while True:
            history = read(run_dir/'history.json') if (run_dir/'history.json').exists() else []
            for row in history:
                step = row['step']
                for path in sorted((run_dir/'inner'/f'step-{step:06d}').glob('task-*.json')):
                    key = (step, path.name)
                    if key not in tasks_seen:
                        fit = read(path)
                        emit('task_complete', task_event(step, fit['mouse'], fit))
                        tasks_seen.add(key)
                if 'query_refit' in row and step not in evaluations_seen:
                    reports = {role: read(run_dir/role/f'step-{step:06d}.json')
                               for role in ('query', 'valid', 'test') if (run_dir/role/f'step-{step:06d}.json').exists()}
                    emit('evaluation_started', dict(step=step))
                    for mouse in config['targets']:
                        emit('refit_complete', refit_event(step, mouse, reports))
                    emit('evaluation_step', dict(step=step, **{role: report['aggregate'] for role, report in reports.items()}))
                    evaluations_seen.add(step)
                if step not in steps_seen:
                    emit('outer_step', row)
                    steps_seen.add(step)
            status = read(run_dir/'status.json')
            selection = read(run_dir/'selection.json') if (run_dir/'selection.json').exists() else None
            if selection is not None and selection['step'] in steps_seen and selection['step'] != selected_seen:
                emit('checkpoint_selected', dict(step=selection['step'], role=selection['role'], loss=selection['aggregate']['loss']))
                selected_seen = selection['step']
            active = status.get('active_step')
            if active is not None:
                for path in sorted((run_dir/'inner'/f'step-{active:06d}').glob('task-*.json')):
                    key = (active, path.name)
                    if key not in tasks_seen:
                        fit = read(path)
                        emit('task_complete', task_event(active, fit['mouse'], fit))
                        tasks_seen.add(key)
            result = read(run_dir/'result.json') if (run_dir/'result.json').exists() else None
            if status['status'] == 'complete':
                emit('training_complete', dict(completed_steps=status['step'], selected_step=selection['step'],
                                              role=selection['role'], loss=selection['aggregate']['loss']))
                break
            if status['status'] == 'interrupted' or (result and result['status'] == 'crash'):
                failure = result if result and result['status'] == 'crash' else status
                emit('training_error', dict(error=failure.get('error', 'Interrupted'),
                    message=failure.get('message', failure.get('reason', 'Training interrupted; inspect original train.log.'))))
                break
            if not follow:
                break
            time.sleep(2)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--follow', action='store_true')
    args = parser.parse_args()
    destination = args.output or args.run_dir/'train.readable.log'
    print(f'Readable log: {destination}', flush=True)
    render(args.run_dir, destination, follow=args.follow)
