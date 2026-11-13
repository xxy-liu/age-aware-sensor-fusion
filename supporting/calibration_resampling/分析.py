"""Calibration-size and block-resampling sensitivity with fixed evaluation data.

The analysis uses saved predictions and separates protocol definition,
calibration, evaluation and summary calculation.
"""
from pathlib import Path
import argparse, hashlib, itertools, json, math, platform, sys, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SRC = ROOT / '03_实验记录'
CACHE = OUT / '缓存'
TARGETS = np.array([.05, .10, .15, .20])
MODES = ['aware', 'zero', 'blind']
POLICIES = ['global', 'age_interpolated']
REPS = 1000
BLOCK = 50  # 50 consecutive endpoints, spaced 0.2 s apart.
FRACTIONS = [.25, .5, 1.]
DRAW_SEED = 20261112


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                     allow_nan=False)+'\n', encoding='utf-8')


def read(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def configurations():
    split_plan = json.loads((SRC/'人群复核/方案.json').read_text(encoding='utf-8'))
    out = []
    for ds in ['REALDISP', 'MHEALTH']:
        real = ds == 'REALDISP'
        seeds = [9101, 9102, 9103] if real else [9301, 9302, 9303]
        for split in range(4):
            if split == 0:
                folder = '收尾实验' if real else '前瞻验证'
                cal = [9, 10, 11] if real else [8, 10]
                test = [12, 13, 14, 15, 16, 17] if real else [2, 4, 7, 9]
                cal_name = 'calibration_REALDISP_CNN_{seed}.npz' if real else 'calibration_{seed}.npz'
                test_name = 'evaluate_REALDISP_CNN_{seed}.npz' if real else 'test_{seed}.npz'
            else:
                sp = split_plan['datasets'][ds]['splits'][split-1]
                folder = f'人群复核/{ds}/划分{split}'
                cal, test = sorted(sp['calibration']), sorted(sp['test'])
                cal_name, test_name = 'calibration_{seed}.npz', 'test_{seed}.npz'
            names = ['aware_native', 'aware_zero', 'blind_native'] if real and split == 0 else ['aware20_native', 'aware20_zero', 'blind20_native']
            out.append(dict(id=f'{ds}_{split}', dataset=ds, split=split,
                            folder=folder, seeds=seeds, cal_people=cal, test_people=test,
                            cal_name=cal_name, test_name=test_name, names=names))
    return out


def source_path(cfg, seed, role):
    return SRC/cfg['folder']/'预测'/cfg[f'{role}_name'].format(seed=seed)


def load(cfg, seed, role, numeric=True):
    with np.load(source_path(cfg, seed, role), allow_pickle=False) as z:
        fields = ['subject', 'row', 'y', 'scenario', 'direction', 'age']
        if 'record' in z.files:
            fields.append('record')
        p = {k: z[k] for k in fields}
        p['scenarios'] = z['scenarios'].tolist()
        if numeric:
            cols = [z['components'].tolist().index(n) for n in cfg['names']]
            p['margin'], p['action'] = z['margin'][:, cols], z['action'][:, cols]
            assert np.isfinite(p['margin']).all()
    assert sorted(np.unique(p['subject']).tolist()) == sorted(cfg[f'{role}_people'])
    return p


def base_indices(p):
    return np.flatnonzero((p['scenario'] == 0) & (p['direction'] == 0))


def blocks(p, base, sid):
    # Circular moving blocks within uninterrupted background runs. Every original
    # background endpoint is one equally likely starting point. Wrapping within
    # its own run retains short runs instead of filtering transition-adjacent data.
    ix = np.flatnonzero((p['subject'][base] == sid) & (p['y'][base] == 0))
    row = p['row'][base[ix]]
    breaks = np.diff(row) != 10
    if 'record' in p:
        breaks |= np.diff(p['record'][base[ix]]) != 0
    bounds = np.r_[0, np.flatnonzero(breaks)+1, len(ix)]
    chunks = []
    for start, stop in zip(bounds[:-1], bounds[1:]):
        n = stop-start
        if n:
            positions=(np.arange(n)[:,None]+np.arange(BLOCK)[None,:]) % n
            chunks.append(ix[start+positions])
    assert chunks, ('No calibration background blocks', sid)
    return np.concatenate(chunks)


def freeze():
    assert not (OUT/'方案.json').exists(), 'Existing frozen protocol; do not overwrite.'
    CACHE.mkdir(exist_ok=True)
    cfgs = configurations()
    sources = {}
    inventories = {}
    # Calibration labels/indices establish feasible budgets; evaluation numeric arrays
    # are not opened here. Checksums read opaque bytes for provenance only.
    for ci, cfg in enumerate(cfgs):
        for seed in cfg['seeds']:
            for role in ['cal', 'test']:
                path = source_path(cfg, seed, role)
                sources[str(path.relative_to(ROOT))] = sha(path)
        p = load(cfg, cfg['seeds'][0], 'cal', numeric=False)
        base = base_indices(p)
        bs = {sid: blocks(p, base, sid) for sid in cfg['cal_people']}
        divisor = math.lcm(*range(1, len(bs)+1))*4
        total_blocks = (min(len(b) for b in bs.values())//BLOCK)//divisor*divisor
        assert total_blocks >= divisor
        saved = {'base_rows': p['row'][base], 'base_subjects': p['subject'][base],
                 'base_labels': p['y'][base]}
        for sid, b in bs.items():
            rng = np.random.default_rng(np.random.SeedSequence([DRAW_SEED, ci, sid]))
            saved[f'blocks_{sid}'] = b
            # Reused for all models, seeds, directions, targets, compositions and
            # nested budgets. These are Monte Carlo draws, not extra participants.
            saved[f'draws_{sid}'] = rng.integers(len(b), size=(REPS, total_blocks), dtype=np.int32)
        np.savez_compressed(CACHE/f'{cfg["id"]}_抽样.npz', **saved)
        inventories[cfg['id']] = dict(total_blocks=total_blocks,
            background_endpoints_per_anchor=total_blocks*BLOCK,
            circular_block_starts_per_person={str(s): len(b) for s,b in bs.items()},
            background_endpoints_per_person={str(s): int(((p['subject'][base]==s)&(p['y'][base]==0)).sum()) for s in bs})
    protected = [ROOT/'outputs/论文'/n for n in
        ['数据年龄感知的传感器融合：阈值效应与校准迁移.tex', '数据年龄感知的传感器融合：阈值效应与校准迁移.pdf']]
    plan = dict(status='FROZEN_POST_EXPOSURE_CALIBRATION_SENSITIVITY',
        date='2026-11-12', utc=datetime.now(timezone.utc).isoformat(),
        authorization='Calibration-size and block-resampling analyses with fixed evaluation participants.',
        question='固定模型和评价参与者时，校准人数、组成与记录预算如何影响迁移后的FPR和CAR？',
        scope='Saved 20-joint-epoch CNN predictions; original allocations plus three dependent repartitions, reported separately; 1000 ms evaluation.',
        primary='Original allocations, aware age-interpolated policy, target 15%; all four targets, all three readouts and both policies retained.',
        configurations=cfgs, targets=TARGETS.tolist(), modes=MODES, policies=POLICIES,
        group_rule='All nonempty subsets of the original calibration participants are evaluated within each partition; evaluation participants remain fixed.',
        full_reference='Thresholds for each calibration subset use all its background endpoints with equal-participant and equal-anchor weights.',
        resampling=dict(repetitions=REPS, seed=DRAW_SEED, block_endpoints=BLOCK,
            endpoint_spacing_seconds=.2, block_nominal_seconds=10,
            rule='Circular moving blocks of 50 endpoints within each uninterrupted background run. Starting endpoints are sampled uniformly over all background endpoints. Wrap only within the same run, including short runs; never connect different runs, records or action intervals. Each original background endpoint has the same expected sampling weight.',
            budget='Common total number of background endpoints across all group sizes/compositions within each partition. Largest block count is floor(minimum background count / 50), rounded down to a multiple of 4*lcm(1,...,K).',
            fractions=FRACTIONS, draws='With replacement within each selected participant; equal block counts per person; paired nested draws across budgets; shared draws across all fitted pipelines.',
            anchors='Four timing anchors contribute equally to global thresholds. Age anchors use all query ages of the selected calibration people and remain fixed across that group\'s record resamples.',
            excluded='No background endpoint is excluded from the resampling population. Repeated points from circular short runs are resampling weights, not independent observations.'),
        uncertainty='2.5/97.5 percentiles describe conditional variation across 1000 draws and equally weighted exact calibration subsets. Evaluation people and fitted models stay fixed. These are not population confidence intervals, fresh cohorts, or independent replications.',
        aggregation='The three seeds and two directions are averaged within each fixed evaluation participant, followed by equal averaging across participants. Metrics are FPR, CAR, signed target gap, participant mean absolute target deviation and worst participant FPR.',
        interpretation='This controlled sensitivity analysis separates record budget from calibration-group composition/size within observed pools; it does not causally identify population shift or guarantee monotonic improvement.',
        inventory=inventories, source_sha256=sources,
        protected_chinese_sha256={str(p.relative_to(ROOT)):sha(p) for p in protected},
        analysis_code_sha256_at_freeze=sha(__file__),
        sampling_sha256={p.name:sha(p) for p in sorted(CACHE.glob('*_抽样.npz'))},
        environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__))
    dump('方案.json', plan)
    print(json.dumps(dict(status=plan['status'], inventory=inventories), ensure_ascii=False), flush=True)


def designs(cfg, inventory):
    rows = []
    for k in range(1, len(cfg['cal_people'])+1):
        for group in itertools.combinations(cfg['cal_people'], k):
            rows.append(dict(group=list(group), k=k, kind='full', fraction=1., start=len(rows), count=1))
    # Offsets are row indices into the dense replicate arrays.
    offset = len(rows)
    for k in range(1, len(cfg['cal_people'])+1):
        for group in itertools.combinations(cfg['cal_people'], k):
            for fraction in FRACTIONS:
                rows.append(dict(group=list(group), k=k, kind='block', fraction=fraction,
                                 start=offset, count=REPS))
                offset += REPS
    return rows, offset


def weighted_thresholds(values, weights):
    order = np.argsort(values, kind='stable')
    s, w = values[order], weights[order]
    val, first = np.unique(s, return_index=True)
    mass = np.add.reduceat(w, first)
    tail = np.r_[np.cumsum(mass[:0:-1])[::-1], 0.]
    return np.array([val[np.flatnonzero(tail <= a+1e-12)[0]] for a in TARGETS])


def empirical_quantiles(sample):
    # With an equal budget per person and anchor every sampled score has equal
    # weight. For strict score > threshold, this order statistic is the smallest
    # observed score whose upper tail is <= alpha (whole ties are rejected).
    n = sample.shape[1]
    ranks = n-np.floor(TARGETS*n+1e-9).astype(int)-1
    sample.partition(ranks.tolist(), axis=1)
    return sample[:, ranks]


def calibrate_one(cfg, plan):
    ident = cfg['id']
    samplefile = CACHE/f'{ident}_抽样.npz'
    assert sha(samplefile) == plan['sampling_sha256'][samplefile.name]
    with np.load(samplefile, allow_pickle=False) as z:
        sample = dict(z)
    inventory = plan['inventory'][ident]
    design, nrows = designs(cfg, inventory)
    # Last axis: global, 500-ms anchor, 1000-ms anchor. Only these local knots
    # are needed for the prespecified 1000-ms evaluation.
    thresholds = np.empty((nrows, 6, 3, 4, 3), dtype=np.float64)
    anchors = np.empty((nrows, 4), dtype=np.float64)
    for si, seed in enumerate(cfg['seeds']):
        p = load(cfg, seed, 'cal')
        base = base_indices(p)
        for key, field in [('base_rows','row'),('base_subjects','subject'),('base_labels','y')]:
            assert np.array_equal(sample[key], p[field][base])
        subjects, labels = p['subject'][base], p['y'][base]
        for dr in (0, 1):
            cells = [np.flatnonzero((p['direction']==dr)&(p['scenario']==a)) for a in range(4)]
            for ix in cells:
                for field in ['row','subject','y']:
                    assert np.array_equal(p[field][ix], p[field][base])
            vals = np.stack([p['margin'][ix] for ix in cells], axis=1)  # endpoint, anchor, mode
            ages = np.stack([p['age'][ix] for ix in cells], axis=1)
            for d in design:
                group, st, count = d['group'], d['start'], d['count']
                group_anchors = np.mean([ages[subjects==s].mean(axis=0, dtype=np.float64) for s in group], axis=0)
                assert np.all(np.diff(group_anchors)>0)
                if si == 0 and dr == 0:
                    anchors[st:st+count] = group_anchors
                else:
                    assert np.allclose(anchors[st:st+count], group_anchors, atol=1e-12, rtol=0)
                if d['kind'] == 'full':
                    ix = np.flatnonzero(np.isin(subjects, group)&(labels==0))
                    w = np.zeros(len(ix))
                    for s in group:
                        take = subjects[ix]==s
                        w[take] = 1/(len(group)*take.sum())
                    for m in range(3):
                        thresholds[st,si*2+dr,m,:,0] = weighted_thresholds(vals[ix,:,m].ravel(),np.repeat(w/4,4))
                        for j,a in enumerate([2,3],1):
                            thresholds[st,si*2+dr,m,:,j] = weighted_thresholds(vals[ix,a,m],w)
                    continue
                nper = int(inventory['total_blocks']*d['fraction'])//len(group)
                assert nper*len(group) == int(inventory['total_blocks']*d['fraction'])
                for begin in range(0, REPS, 64):
                    end = min(begin+64, REPS)
                    ix = np.concatenate([sample[f'blocks_{s}'][sample[f'draws_{s}'][begin:end,:nper]].reshape(end-begin,-1) for s in group], axis=1)
                    for m in range(3):
                        thresholds[st+begin:st+end,si*2+dr,m,:,0] = empirical_quantiles(vals[ix,:,m].reshape(end-begin,-1))
                        for j,a in enumerate([2,3],1):
                            thresholds[st+begin:st+end,si*2+dr,m,:,j] = empirical_quantiles(vals[ix,a,m])
            print('CALIBRATION',ident,seed,dr,flush=True)
    np.savez_compressed(CACHE/f'{ident}_阈值.npz', thresholds=thresholds, anchors=anchors)
    dump(f'缓存/{ident}_设计.json', design)
    return nrows


def calibrate():
    plan = read('方案.json')
    digest=sha(__file__)
    if digest != plan['analysis_code_sha256_at_freeze']:
        repair=json.loads((OUT/'修复记录.json').read_text(encoding='utf-8-sig'))
        assert any(r['from_sha256']==plan['analysis_code_sha256_at_freeze'] and r['to_sha256']==digest for r in repair.get('code_repairs',[])), 'Unrecorded code change.'
    start = time.time()
    for cfg in plan['configurations']:
        result = CACHE/f'{cfg["id"]}_阈值.npz'
        if result.exists():
            print('EXISTING',cfg['id'],flush=True)
            continue
        n = calibrate_one(cfg, plan)
        print('CALIBRATED',cfg['id'],n,round(time.time()-start,1),'seconds',flush=True)
    dump('阈值冻结.json', dict(utc=datetime.now(timezone.utc).isoformat(),
        plan_sha256=sha(OUT/'方案.json'), code_sha256=sha(__file__),
        evaluation_numeric_access_in_calibration_phase=False, previously_exposed_predictions=True,
        files_sha256={p.name:sha(p) for p in sorted(CACHE.glob('*_阈值.npz'))},
        design_sha256={p.name:sha(p) for p in sorted(CACHE.glob('*_设计.json'))}))


def accepted(sorted_values, threshold):
    return len(sorted_values)-np.searchsorted(sorted_values, threshold, side='right')


def interpolated(th, anchors, age):
    # All evaluation ages at nominal 1000 ms lie above the 500-ms knot.
    assert np.all(age >= anchors[:,2])
    tau = th[:,:,1]+((th[:,:,2]-th[:,:,1])/(anchors[:,3]-anchors[:,2])[:,None])*(age-anchors[:,2])[:,None]
    above = age >= anchors[:,3]
    tau[above] = th[above,:,2]
    return tau


def evaluate():
    plan, frozen = read('方案.json'), read('阈值冻结.json')
    assert sha(OUT/'方案.json') == frozen['plan_sha256']
    for cfg in plan['configurations']:
        ident = cfg['id']
        path = CACHE/f'{ident}_阈值.npz'
        assert sha(path) == frozen['files_sha256'][path.name]
        with np.load(path, allow_pickle=False) as z:
            th, anchors = z['thresholds'], z['anchors']
        people = cfg['test_people']
        # Repetition, mode, policy, target, fixed person, metric(FPR,CAR).
        rates = np.zeros((len(th),3,2,4,len(people),2), dtype=np.float64)
        for si, seed in enumerate(cfg['seeds']):
            p = load(cfg, seed, 'test')
            for dr in (0,1):
                for pi,sid in enumerate(people):
                    ix=np.flatnonzero((p['scenario']==3)&(p['direction']==dr)&(p['subject']==sid))
                    assert ix.size
                    y, age = p['y'][ix],p['age'][ix]
                    nb,na=(y==0).sum(),(y>0).sum()
                    assert nb and na
                    for m in range(3):
                        score, action = p['margin'][ix,m],p['action'][ix,m]
                        t = th[:,si*2+dr,m]
                        good=(y>0)&(action==y)
                        rates[:,m,0,:,pi,0] += accepted(np.sort(score[y==0]),t[:,:,0])/nb/6
                        rates[:,m,0,:,pi,1] += accepted(np.sort(score[good]),t[:,:,0])/na/6
                        for a in np.unique(age):
                            tau=interpolated(t,anchors,float(a))
                            same=age==a
                            rates[:,m,1,:,pi,0] += accepted(np.sort(score[same&(y==0)]),tau)/nb/6
                            rates[:,m,1,:,pi,1] += accepted(np.sort(score[same&good]),tau)/na/6
            print('EVALUATION',ident,seed,flush=True)
        assert np.isfinite(rates).all() and rates.min()>=0 and rates.max()<=1+1e-12
        np.savez_compressed(CACHE/f'{ident}_逐人.npz',rates=rates,subjects=np.array(people))
    dump('完成.json',dict(status='COMPUTED_PENDING_AUDIT',utc=datetime.now(timezone.utc).isoformat(),
        plan_sha256=sha(OUT/'方案.json'),code_sha256=sha(__file__),
        output_sha256={p.name:sha(p) for p in sorted(CACHE.glob('*.npz'))}))


def summarize():
    plan=read('方案.json')
    group_rows, size_rows=[] ,[]
    for cfg in plan['configurations']:
        ident=cfg['id']; design=read(f'缓存/{ident}_设计.json')
        with np.load(CACHE/f'{ident}_逐人.npz',allow_pickle=False) as z:
            rates=z['rates']
        for scope in ['group','size']:
            keys=[]
            for d in design:
                key=(d['kind'],d['k'],d['fraction'],tuple(d['group']) if scope=='group' else ())
                if key not in keys: keys.append(key)
            for kind,k,fraction,group in keys:
                chosen=[d for d in design if (d['kind'],d['k'],d['fraction'])==(kind,k,fraction) and (not group or tuple(d['group'])==group)]
                v=np.concatenate([rates[d['start']:d['start']+d['count']] for d in chosen])
                for m,mode in enumerate(MODES):
                    for po,policy in enumerate(POLICIES):
                        for ti,target in enumerate(TARGETS):
                            fp=v[:,m,po,ti,:,0]; car=v[:,m,po,ti,:,1]
                            fmean=fp.mean(axis=1)
                            lo,hi=np.quantile(fmean,[.025,.975])
                            row=dict(dataset=cfg['dataset'],split=cfg['split'],kind=kind,k=k,
                                fraction=fraction,group=','.join(map(str,group)),combinations=len(chosen),
                                replicates_per_combination=1 if kind=='full' else REPS,
                                budget_endpoints_per_anchor=(plan['inventory'][ident]['background_endpoints_per_anchor']*fraction if kind=='block' else np.nan),
                                mode=mode,policy=policy,target=target,
                                mean_fpr=fmean.mean(),conditional_low=lo,conditional_high=hi,
                                min_fpr=fmean.min(),max_fpr=fmean.max(),mean_car=car.mean(),
                                mean_signed_gap=fmean.mean()-target,
                                mean_absolute_mean_gap=np.abs(fmean-target).mean(),
                                mean_participant_absolute_gap=np.abs(fp-target).mean(),
                                mean_worst_participant_fpr=fp.max(axis=1).mean())
                            (group_rows if scope=='group' else size_rows).append(row)
    for name,rows in [('组合.csv',group_rows),('汇总.csv',size_rows)]:
        pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig',float_format='%.17g')
    f=pd.DataFrame(size_rows)
    print(f[(f.split==0)&(f.kind=='block')&(f['mode']=='aware')&(f.policy=='age_interpolated')&(f.target==.15)][['dataset','k','fraction','mean_fpr','conditional_low','conditional_high','mean_car','mean_participant_absolute_gap']].to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['freeze','calibrate','evaluate','summarize'])
    {'freeze':freeze,'calibrate':calibrate,'evaluate':evaluate,'summarize':summarize}[parser.parse_args().phase]()
