"""Replay the two frozen supplementary analyses using packaged caches."""
from pathlib import Path
import argparse
import importlib.util
import hashlib
import json
import shutil
import sys
import time

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replay_plan(source, out):
    """Validate editorial changes and prepare the plan used for this replay."""
    revision = read(BASE/'revision.json')
    for name in ['分析.py', '方案.json']:
        path = source/name
        relative = path.relative_to(BASE).as_posix()
        if sha(path) != revision['files'][relative]['after_sha256']:
            raise ValueError(f'Publication source changed: {relative}')
    source_plan = read(source/'方案.json')
    source_freeze = read(source/'阈值冻结.json')
    code_entry = revision['files'][(source/'分析.py').relative_to(BASE).as_posix()]
    plan_entry = revision['files'][(source/'方案.json').relative_to(BASE).as_posix()]
    code_key = 'analysis_code_sha256' if source.name == 'cache_analyses' else 'code_sha256'
    if source_freeze[code_key] != code_entry['before_sha256']:
        raise ValueError('Original frozen code hash differs from the revision record')
    if source_freeze['plan_sha256'] != plan_entry['before_sha256']:
        raise ValueError('Original frozen plan hash differs from the revision record')
    if source.name == 'cache_analyses':
        if source_plan['analysis_code_sha256'] != code_entry['before_sha256']:
            raise ValueError('Original plan code hash differs from the revision record')
        source_plan['analysis_code_sha256'] = code_entry['after_sha256']
    (out/'方案.json').write_text(json.dumps(source_plan, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return dict(version=revision['version'],
                source_plan_sha256=sha(source/'方案.json'),
                source_code_sha256=sha(source/'分析.py'),
                original_frozen_record_sha256=sha(source/'阈值冻结.json'),
                original_plan_sha256=source_freeze['plan_sha256'],
                original_code_sha256=source_freeze[code_key])


def module_at(path):
    spec = importlib.util.spec_from_file_location('frozen_analysis', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare_csv(actual, reference):
    a = pd.read_csv(actual, float_precision='round_trip')
    b = pd.read_csv(reference, float_precision='round_trip')
    if list(a.columns) != list(b.columns) or a.shape != b.shape:
        raise ValueError(f'Summary layout changed: {actual.name}')
    maximum = 0.
    for name in a:
        if pd.api.types.is_numeric_dtype(a[name]) and pd.api.types.is_numeric_dtype(b[name]):
            x, y = a[name].to_numpy(dtype=float), b[name].to_numpy(dtype=float)
            if not np.array_equal(np.isnan(x), np.isnan(y)):
                raise ValueError(f'Missing-value pattern changed: {name}')
            finite = np.isfinite(x) & np.isfinite(y)
            if finite.any():
                maximum = max(maximum,float(np.max(np.abs(x[finite]-y[finite]))))
            if not np.allclose(x,y,atol=2e-12,rtol=0,equal_nan=True):
                raise ValueError(f'Numeric replay mismatch: {actual.name}, {name}')
        elif not a[name].fillna('').equals(b[name].fillna('')):
            raise ValueError(f'Row labels changed: {name}')
    return dict(file=actual.name, rows=len(a), maximum_absolute_error=maximum)


def lookup_groups():
    config=read(BASE/'config/evaluation.json')
    return {(g['dataset'], 0 if g['partition']=='O' else int(g['partition'])):g
            for g in config['groups'] if g['architecture']=='CNN'}


def cached_queries(groups, dataset, split, seed, role, names, numeric=True):
    group=groups[(dataset,split)]
    run=next(r for r in group['runs'] if r['seed']==seed)
    key='calibration' if role=='cal' else 'evaluation'
    with np.load(BASE/run[key],allow_pickle=False) as z:
        fields=['subject','row','old','y','scenario','direction','age']
        if 'record' in z.files:
            fields.append('record')
        p={k:z[k] for k in fields}
        p['scenarios']=z['scenarios'].tolist()
        if numeric:
            cols=[z['components'].tolist().index(n) for n in names]
            p['margin'],p['action']=z['margin'][:,cols],z['action'][:,cols]
    if sorted(np.unique(p['subject']).tolist()) != sorted(group[key]):
        raise ValueError('Participant allocation mismatch')
    return p


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('analysis',choices=['cache','resampling'])
    args=parser.parse_args()
    started=time.perf_counter()
    name='cache_analyses' if args.analysis=='cache' else 'calibration_resampling'
    source=BASE/'supporting'/name
    out=BASE/'outputs'/name
    out.mkdir(parents=True,exist_ok=True)
    verification=out/'replay_verification.json'
    if verification.exists():
        verification.unlink()
    frozen=module_at(source/'分析.py')
    frozen.OUT=out
    groups=lookup_groups()
    # Adapt paths and editorial metadata in replay copies; preserve the original
    # audit records and all numerical functions, inputs and study-design values.
    metadata = replay_plan(source, out)
    if args.analysis=='cache':
        def loadpack(cfg,seed,role):
            dataset='REALDISP' if cfg['k']==34 else 'MHEALTH'
            return cached_queries(groups,dataset,0,seed,role,cfg['names'])
        frozen.loadpack=loadpack
        frozen.calibrate()
        if read(out/'阈值.json') != read(source/'阈值.json'):
            raise ValueError('Recomputed target-sweep thresholds changed')
        frozen.evaluate()
        targets=sorted(source.glob('*.csv'))
    else:
        cache=out/'缓存'
        cache.mkdir(exist_ok=True)
        frozen.CACHE=cache
        replay_freeze = read(source/'阈值冻结.json')
        replay_freeze['plan_sha256'] = sha(out/'方案.json')
        replay_freeze['code_sha256'] = sha(source/'分析.py')
        replay_freeze['replay_metadata_adaptation'] = metadata
        (out/'阈值冻结.json').write_text(json.dumps(replay_freeze, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        for path in (source/'缓存').iterdir():
            if path.name.endswith(('_阈值.npz','_设计.json')):
                shutil.copyfile(path,cache/path.name)
        def load(cfg,seed,role,numeric=True):
            return cached_queries(groups,cfg['dataset'],cfg['split'],seed,role,cfg['names'],numeric)
        frozen.load=load
        # Apply the saved block-resample thresholds to the fixed evaluation
        # predictions again. Sampling and threshold fitting are not rerun.
        frozen.evaluate()
        frozen.summarize()
        targets=[source/'汇总.csv',source/'组合.csv']
    checks=[compare_csv(out/path.name,path) for path in targets]
    result=dict(status='PASS',analysis=args.analysis,checks=checks,metadata_revision=metadata,
                elapsed_seconds=time.perf_counter()-started,
                scope=('Recomputed calibration thresholds, query changes, target transfer and crossed readouts from prediction caches.'
                       if args.analysis=='cache' else
                       'Recomputed evaluation rates and summaries from the frozen resampling thresholds; no new block draws or threshold fitting.'))
    verification.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('PASS',args.analysis,'replay;',len(checks),'tables verified',flush=True)


if __name__=='__main__':
    main()
