"""Portable CPU evaluation of the manuscript's saved prediction caches."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import platform
import sys
import time

import numpy as np

BASE = Path(__file__).resolve().parent
METRICS = ['macro_f1', 'car', 'fpr', 'miss', 'wrong_action', 'accuracy']
LEGACY = dict(car='delivered_correct', wrong_action='other_action_error')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_files():
    manifest = read(BASE/'manifest.json')
    for relative, value in manifest['files'].items():
        if sha(BASE/relative) != value['sha256']:
            raise ValueError(f'Package file changed: {relative}')
    return len(manifest['files'])


def load_cache(run, role, group, selected=None):
    with np.load(BASE/run[role], allow_pickle=False) as z:
        names = z['components'].tolist()
        cols = list(range(len(names))) if selected is None else [names.index(x) for x in selected]
        out = {k: z[k] for k in ['subject','row','y','scenario','direction','age']}
        out['margin'] = z['margin'][:, cols]
        out['action'] = z['action'][:, cols]
        out['components'] = [names[j] for j in cols]
        out['scenarios'] = z['scenarios'].tolist()
    expected = sorted(group['calibration' if role == 'calibration' else 'evaluation'])
    if sorted(np.unique(out['subject']).tolist()) != expected:
        raise ValueError(f'Participant allocation mismatch: {run[role]}')
    n = len(out['y'])
    if any(len(out[k]) != n for k in ['subject','row','scenario','direction','age','margin','action']):
        raise ValueError('Unequal query-array lengths')
    if not np.isfinite(out['margin']).all() or not np.isfinite(out['age']).all():
        raise ValueError('Non-finite scores or ages')
    if not ((out['y'] >= 0) & (out['y'] < group['classes'])).all():
        raise ValueError('Label outside fixed class set')
    if not ((out['action'] > 0) & (out['action'] < group['classes'])).all():
        raise ValueError('Action outside fixed class set')
    if set(np.unique(out['direction'])) != {0,1}:
        raise ValueError('Expected two sensor directions')
    return out


def metrics(y, pred, classes):
    cm = np.bincount(y*classes+pred, minlength=classes*classes).reshape(classes, classes)
    den = cm.sum(0)+cm.sum(1)
    action = y > 0
    if not action.any() or not (~action).any():
        raise ValueError('Both background and action queries are required')
    return dict(n=len(y), macro_f1=float(np.divide(2*cm.diagonal(), den,
                out=np.zeros(classes), where=den>0).mean()),
                car=float((pred[action] == y[action]).mean()),
                fpr=float((pred[~action] > 0).mean()),
                miss=float((pred[action] == 0).mean()),
                wrong_action=float(((pred[action] > 0) & (pred[action] != y[action])).mean()),
                accuracy=float((pred == y).mean()))


def curve(y, scores, actions):
    order = np.argsort(-scores, kind='stable')
    s = scores[order]
    last = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s)-1]
    background = y[order] == 0
    good = (y[order] > 0) & (actions[order] == y[order])
    x = np.r_[0., np.cumsum(background)[last]/background.sum()]
    v = np.r_[0., np.cumsum(good)[last]/(y > 0).sum()]
    small = x < .2
    xx, yy = np.r_[x[small], .2], np.r_[v[small], np.interp(.2, x, v)]
    integrate = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    return {**{f'correct_at_{t:02d}': float(np.interp(t/100, x, v)) for t in [5,10,15,20]},
            'partial_area_20': float(integrate(yy, xx)/.2),
            'rank_accuracy': float((actions[y > 0] == y[y > 0]).mean())}


def weighted_threshold(scores, weights, target):
    active = weights > 0
    s, w = scores[active], weights[active]
    if target >= 1:
        return float(np.nextafter(s.min(), -np.inf))
    order = np.argsort(s, kind='stable')
    s, w = s[order], w[order]
    values, first = np.unique(s, return_index=True)
    mass = np.add.reduceat(w, first)
    tail = np.r_[np.cumsum(mass[:0:-1])[::-1], 0.]
    return float(values[np.flatnonzero(tail <= target+1e-12)[0]])


def verify_table2_thresholds(cache, saved, group):
    """Reconstruct matching from calibration labels only, before evaluation."""
    checked = []
    si = cache['scenarios'].index('1000')
    for direction in [0,1]:
        take = (cache['direction'] == direction) & (cache['scenario'] == si)
        s, y, people = cache['margin'][take], cache['y'][take], cache['subject'][take]
        w = np.zeros(len(y))
        for sid in group['calibration']:
            bg = (people == sid) & (y == 0)
            if not bg.any():
                raise ValueError('Calibration participant without background queries')
            w[bg] = 1/(len(group['calibration'])*bg.sum())
        target = float(w[s[:,0] > 0].sum())
        actual = weighted_threshold(s[:,1], w, target)
        pol = saved[str(direction)]['components'][cache['components'][1]]
        expected = pol['local_match'][si]
        # This check is exact: the selected observed score is unchanged.
        if actual != expected or abs(target-pol['local_targets'][si]) > 1e-12:
            raise ValueError(f'Calibration threshold differs: {actual}, {expected}')
        achieved = float(w[s[:,1] > actual].sum())
        if achieved > target+1e-12:
            raise ValueError('Calibration target is not met')
        checked.append(dict(direction=direction, target=target, threshold=actual, achieved=achieved))
    return checked


def table2(config, policies, out):
    detailed, calibration_checks = [], []
    for group in config['groups']:
        if group['partition'] != 'O' or group['architecture'] != 'CNN':
            continue
        names = ['aware_native','aware_zero'] if group['dataset'] == 'REALDISP' else ['aware20_native','aware20_zero']
        for run in group['runs']:
            saved = policies[run['id']]
            cal = load_cache(run, 'calibration', group, names)
            checked = verify_table2_thresholds(cal, saved, group)
            calibration_checks.append(dict(run=run['id'], checks=checked))
            del cal
            p = load_cache(run, 'evaluation', group, names)
            si = p['scenarios'].index('1000')
            for dr in [0,1]:
                threshold = saved[str(dr)]['components'][names[1]]['local_match'][si]
                for sid in group['evaluation']:
                    take = (p['scenario'] == si) & (p['direction'] == dr) & (p['subject'] == sid)
                    for label, j, t in [('Native',0,0.),('Zero',1,0.),('Zero+match',1,threshold)]:
                        pred = np.where(p['margin'][take,j] > t, p['action'][take,j], 0)
                        detailed.append(dict(dataset=group['dataset'], seed=run['seed'], direction=dr,
                                             subject=sid, readout=label, **metrics(p['y'][take],pred,group['classes'])))
            del p
            print('EVALUATED', run['id'], flush=True)
    people, summary = [], []
    expected_rows = rows(BASE/'reference/table2.csv')
    for expected in expected_rows:
        subset = [x for x in detailed if x['dataset'] == expected['dataset'] and x['readout'] == expected['readout']]
        participant_rows = []
        for sid in sorted({x['subject'] for x in subset}):
            q = [x for x in subset if x['subject'] == sid]
            if len(q) != 6 or len({(x['seed'],x['direction']) for x in q}) != 6:
                raise ValueError('Incomplete seed/direction pairing')
            record = dict(dataset=expected['dataset'], readout=expected['readout'], subject=sid,
                          **{m: float(np.mean([x[m] for x in q])) for m in METRICS})
            people.append(record)
            participant_rows.append(record)
        record = dict(dataset=expected['dataset'], readout=expected['readout'],
                      **{m: 100*float(np.mean([x[m] for x in participant_rows])) for m in ['macro_f1','car','fpr']})
        for m in ['macro_f1','car','fpr']:
            if f'{record[m]:.2f}' != expected[m]:
                raise ValueError(f'Table II mismatch: {record}, {expected}')
        summary.append(record)
    write_csv(out/'table2.csv', summary)
    write_csv(out/'table2_participants.csv', people)
    write_csv(out/'table2_runs.csv', detailed)
    latex = ['\\begin{tabular}{llrrr}', '\\toprule', 'Dataset & Readout & Macro-F1 & CAR & FPR \\\\', '\\midrule']
    for r in summary:
        latex.append(f"{r['dataset']} & {r['readout']} & {r['macro_f1']:.2f} & {r['car']:.2f} & {r['fpr']:.2f} \\\\")
    latex += ['\\bottomrule', '\\end{tabular}']
    (out/'table2.tex').write_text('\n'.join(latex)+'\n', encoding='utf-8')
    return dict(status='PASS', table=2, displayed_values_matched=18, run_rows=len(detailed),
                participant_rows=len(people), threshold_checks=calibration_checks,
                aggregation=config['aggregation'])


def thresholds_for(policy, scenario, ages):
    result = {'raw': 0., 'global15': policy['global15']}
    if scenario < 4:
        result['local_match'] = policy['local_match'][scenario]
    if 'knots' in policy:
        result.update(global_match=policy['global_match'],
                      interp_match=np.interp(ages, policy['knots'], policy['local_match']),
                      interp15=np.interp(ages, policy['knots'], policy['local15']))
    else:
        result['local15'] = policy['local15'][scenario]
    return result


def check_reference(actual, reference, key_fields, values):
    lookup = {tuple(str(x[k]) for k in key_fields): x for x in reference}
    if len(lookup) != len(reference) or len(actual) != len(reference):
        raise ValueError('Missing or duplicated reference rows')
    maximum = 0.
    for x in actual:
        y = lookup[tuple(str(x[k]) for k in key_fields)]
        if 'n' in x and int(y['n']) != x['n']:
            raise ValueError('Query count changed')
        for m in values:
            err = abs(x[m]-float(y[LEGACY.get(m,m)]))
            maximum = max(maximum, err)
            if err > 2e-12:
                raise ValueError(f'Reference mismatch: {x}, {m}, {err}')
    return maximum


def all_cached_results(config, policies, out):
    report = []
    for group in config['groups']:
        metrics_rows, curve_rows = [], []
        for run in group['runs']:
            p = load_cache(run,'evaluation',group)
            for si, scene in enumerate(p['scenarios']):
                for dr in [0,1]:
                    for sid in group['evaluation']:
                        take = (p['scenario'] == si) & (p['direction'] == dr) & (p['subject'] == sid)
                        y, scores, choices, ages = p['y'][take],p['margin'][take],p['action'][take],p['age'][take]
                        for j, name in enumerate(p['components']):
                            meta = dict(seed=run['seed'], scenario=scene, direction=dr, subject=sid, component=name)
                            pol = policies[run['id']][str(dr)]['components'][name]
                            for policy, t in thresholds_for(pol,si,ages).items():
                                pred = np.where(scores[:,j] > t, choices[:,j], 0)
                                metrics_rows.append(dict(**meta,policy=policy,**metrics(y,pred,group['classes'])))
                            curve_rows.append(dict(**meta,**curve(y,scores[:,j],choices[:,j])))
            del p
        keys = ['seed','scenario','direction','subject','component']
        metric_error = check_reference(metrics_rows,rows(BASE/group['metrics_reference']),keys+['policy'],METRICS)
        curve_metrics = [x for x in curve_rows[0] if x not in keys]
        curve_error = check_reference(curve_rows,rows(BASE/group['curves_reference']),keys,curve_metrics)
        write_csv(out/f"{group['id']}_metrics.csv", metrics_rows)
        write_csv(out/f"{group['id']}_curves.csv", curve_rows)
        report.append(dict(group=group['id'], metrics=len(metrics_rows), curves=len(curve_rows),
                           maximum_absolute_error=max(metric_error,curve_error)))
        print('VERIFIED',group['id'],len(metrics_rows),'metrics',len(curve_rows),'curves',flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--table', type=int, choices=[2], default=2)
    parser.add_argument('--all', action='store_true', help='Also recompute base metrics/curves for all included groups.')
    parser.add_argument('--verify-files', action='store_true', help='Verify every packaged file SHA-256 first.')
    args = parser.parse_args()
    started = time.perf_counter()
    out = BASE/'outputs'
    out.mkdir(exist_ok=True)
    if (out/'verification.json').exists():
        (out/'verification.json').unlink()
    report = dict(python=platform.python_version(), numpy=np.__version__,
                  scope='Computational reproduction from previously saved predictions; no retraining or new inference.')
    if args.verify_files:
        report['package_files_verified'] = verify_files()
    config, policies = read(BASE/'config/evaluation.json'), read(BASE/'config/thresholds.json')
    report['table2'] = table2(config, policies, out)
    if args.all:
        report['all_groups'] = all_cached_results(config, policies, out)
    report.update(status='PASS', elapsed_seconds=time.perf_counter()-started)
    dump(out/'verification.json', report)
    print('PASS: all 18 displayed Table II values reproduced.',flush=True)
    print('Outputs:',str(out),flush=True)


if __name__ == '__main__':
    main()
