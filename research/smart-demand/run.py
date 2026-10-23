"""冻结配置后先开发选型，再对后段六个14天窗口执行同样的滚动预测。"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch

from data import DEFAULT_INPUT, ROOT, load_panel, make_fold, baselines
from models import fit_predict, ridge_predict

HERE = Path(__file__).resolve().parent


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def metric(y, pred):
    mask = np.isfinite(y)
    error = np.asarray(pred, dtype=float)[mask] - y[mask]
    total = float(y[mask].sum())
    return {'n': int(mask.sum()), 'abs_error': float(np.abs(error).sum()),
            'mae': float(np.abs(error).mean()), 'bias': float(error.mean()),
            'wape': float(np.abs(error).sum() / total) if total else None,
            'truth_sum': total, 'pred_observed_sum': float(np.asarray(pred)[mask].sum())}


def score_record(fold, pred, scope, phase, start, name, seed):
    row = {'scope': scope, 'phase': phase, 'start': start, 'method': name, 'seed': seed}
    row.update(metric(fold['truth'], pred))
    row['total14_ae'] = abs(float(np.sum(pred)) - float(np.sum(fold['truth']))) if scope == 'national' else None
    row['eligible_items'] = len(fold['indices'])
    return row


def run(args):
    config = json.loads(args.config.read_text(encoding='utf-8'))
    args.output.mkdir(parents=True, exist_ok=True)
    config_hash = digest(args.config)
    manifest_path = args.output / 'manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        if not args.resume or previous['config_sha256'] != config_hash or previous['input_sha256'] != digest(args.input):
            raise ValueError('输出已有结果；相同输入配置才能使用 --resume，否则请换新目录。')
    manifest = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
                'command': sys.argv, 'config': config, 'config_sha256': config_hash,
                'input': str(args.input.resolve()), 'input_sha256': digest(args.input),
                'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__, 'pandas': pd.__version__}
    write_json(manifest_path, manifest)
    panel = load_panel(args.input)
    records, fit_records, chosen = [], [], {}
    started = time.perf_counter()
    for phase in ('development', 'test'):
        if phase == 'test':
            # 这个文件写出时还没有读取任何测试预测的误差。
            all_dev = pd.DataFrame(records)
            for scope in config['scopes']:
                part = all_dev[(all_dev.scope == scope) & (all_dev.seed == 'ensemble')]
                totals = part.groupby('method')[['abs_error', 'n']].sum()
                totals['mae'] = totals.abs_error / totals.n
                neural = [c['name'] for c in config['candidates']]
                median = [f'median{x}' for x in (14, 28, 56)]
                chosen[scope] = {'neural': str(totals.loc[neural].mae.idxmin()),
                                 'median': str(totals.loc[median].mae.idxmin()),
                                 'development_mae': totals.mae.to_dict()}
            write_json(args.output / 'selection_before_test.json', chosen)
            print('SELECTED BEFORE TEST ' + json.dumps(chosen, ensure_ascii=False), flush=True)
        for scope, settings in config['scopes'].items():
            candidates = config['candidates'] if phase == 'development' else [c for c in config['candidates'] if c['name'] == chosen[scope]['neural']]
            for start in config[phase + '_starts']:
                origin = int(panel['days'].get_loc(pd.Timestamp(start))) - 1
                print(f'{phase} {scope} {start}: prepare', flush=True)
                folds = {}
                for behavior in (False, True):
                    folds[behavior] = make_fold(panel, origin, scope=scope, behavior=behavior, stride=settings['stride'])
                fold = folds[True]
                assert fold['latest_label'] <= origin
                folder = args.output / 'predictions' / scope / phase / start
                folder.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(folder / 'truth.npz', truth=fold['truth'], indices=fold['indices'],
                                    base=fold['current_base'], origin=origin)
                for name, prediction in baselines(panel, origin, scope=scope, indices=fold['indices']).items():
                    np.savez_compressed(folder / (name + '.npz'), prediction=prediction)
                    records.append(score_record(fold, prediction, scope, phase, start, name, 'ensemble'))
                ridge_file = folder / 'ridge.npz'
                ridge_trace_file = folder / 'ridge.json'
                if args.resume and ridge_file.exists() and ridge_trace_file.exists():
                    prediction = np.load(ridge_file)['prediction']
                    trace = json.loads(ridge_trace_file.read_text(encoding='utf-8'))
                else:
                    prediction, trace = ridge_predict(fold['x'], fold['y'], fold['current_x'], alpha=config['ridge_alpha'])
                    np.savez_compressed(ridge_file, prediction=prediction)
                    write_json(ridge_trace_file, trace)
                records.append(score_record(fold, prediction, scope, phase, start, 'ridge', 'ensemble'))
                for candidate in candidates:
                    f = folds[candidate['behavior']]
                    predictions = []
                    for seed in config[phase + '_seeds']:
                        name = candidate['name']
                        pred_file, trace_file = folder / f'{name}_{seed}.npz', folder / f'{name}_{seed}.json'
                        if args.resume and pred_file.exists() and trace_file.exists():
                            prediction = np.load(pred_file)['prediction']
                            trace = json.loads(trace_file.read_text(encoding='utf-8'))
                        else:
                            prediction, trace = fit_predict(f['x'], f['y'], f['base'], f['current_x'], f['current_base'],
                                seed=seed, loss=candidate['loss'], epochs=settings['epochs'], hidden=tuple(settings['hidden']),
                                batch_size=settings['batch_size'], lr=config['learning_rate'], weight_decay=config['weight_decay'])
                            np.savez_compressed(pred_file, prediction=prediction)
                            write_json(trace_file, trace)
                        trace_small = {k: v for k,v in trace.items() if k not in ('scaler_mean','scaler_std')}
                        fit_records.append({'scope': scope, 'phase': phase, 'start': start, 'method': name, 'seed': seed,
                                            'latest_label': str(panel['days'][f['latest_label']].date()), **trace_small})
                        records.append(score_record(f, prediction, scope, phase, start, name, str(seed)))
                        predictions.append(prediction)
                    prediction = np.mean(predictions, axis=0)
                    np.savez_compressed(folder / f'{name}_ensemble.npz', prediction=prediction)
                    records.append(score_record(f, prediction, scope, phase, start, name, 'ensemble'))
                pd.DataFrame(records).to_csv(args.output / 'window_metrics.csv', index=False)
                write_json(args.output / 'fits.json', fit_records)
                print(f'{phase} {scope} {start}: done, train={len(fold["x"])} eligible={len(fold["indices"])} elapsed={time.perf_counter()-started:.1f}s', flush=True)
    manifest.update(status='success', finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter()-started,
                    neural_fits=len(fit_records), source_sha256={p.name: digest(p) for p in (HERE/'data.py', HERE/'models.py', HERE/'run.py')})
    write_json(manifest_path, manifest)
    print('SUCCESS ' + str(args.output), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--config', type=Path, default=HERE/'config.json')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/smart-demand-20261023')
    parser.add_argument('--resume', action='store_true')
    run(parser.parse_args())
