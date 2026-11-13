"""Numerical verification of calibration-size and block-resampling analyses.

Thresholds are reconstructed with sorted weighted CDFs and metrics with direct
query-level predictions, separately from the analysis implementation.
"""
from pathlib import Path
import hashlib, json, math, sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd

sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
CACHE=OUT/'缓存'
plan=json.loads((OUT/'方案.json').read_text(encoding='utf-8'))
freeze=json.loads((OUT/'阈值冻结.json').read_text(encoding='utf-8'))
targets=np.array(plan['targets'])


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def pack(cfg,seed,role):
    path=ROOT/'03_实验记录'/cfg['folder']/'预测'/cfg[f'{role}_name'].format(seed=seed)
    with np.load(path,allow_pickle=False) as z:
        names=z['components'].tolist()
        cols=[names.index(n) for n in cfg['names']]
        p={k:z[k] for k in ['subject','row','y','scenario','direction','age']}
        p['scores']=z['margin'][:,cols]
        p['actions']=z['action'][:,cols]
        if 'record' in z.files: p['record']=z['record']
    return p


def cdf_threshold(score,weight,target):
    order=np.argsort(score,kind='stable')
    ss,ww=score[order],weight[order]
    last=np.r_[np.flatnonzero(ss[1:]!=ss[:-1]),len(ss)-1]
    values=ss[last]
    cumulative=np.cumsum(ww,dtype=np.longdouble)[last]
    j=min(int(np.searchsorted(cumulative,1-target-1e-12,side='left')),len(values)-1)
    # Validate the candidate with pairwise sums instead of relying on accumulated
    # CDF rounding at exact integer percentile boundaries.
    while weight[score>values[j]].sum()>target+1e-12:
        j+=1
    while j and weight[score>values[j-1]].sum()<=target+1e-12:
        j-=1
    value=values[j]
    assert weight[score>value].sum() <= target+1e-12
    if j:
        assert weight[score>values[j-1]].sum() > target-1e-12
    return value


assert sha(OUT/'方案.json')==freeze['plan_sha256']
assert sha(OUT/'分析.py')==freeze['code_sha256']
if freeze['code_sha256']!=plan['analysis_code_sha256_at_freeze']:
    repairs=json.loads((OUT/'修复记录.json').read_text(encoding='utf-8-sig'))['code_repairs']
    assert any(r['from_sha256']==plan['analysis_code_sha256_at_freeze'] and r['to_sha256']==freeze['code_sha256'] for r in repairs)
for path,digest in plan['source_sha256'].items():
    assert sha(ROOT/path)==digest, ('Changed source',path)
for path,digest in plan['protected_chinese_sha256'].items():
    assert sha(ROOT/path)==digest, ('Changed Chinese manuscript',path)
old=json.loads((ROOT/'outputs/补充分析/阈值.json').read_text(encoding='utf-8'))
old_rates=pd.read_csv(ROOT/'outputs/补充分析/校准迁移逐人.csv')
checks=dict(source_files=len(plan['source_sha256']),calibration_values_reconstructed=0,
            direct_evaluation_rates_reconstructed=0,original_reference_rate_matches=0,
            background_endpoints_verified=0)
summary=pd.read_csv(OUT/'汇总.csv')
groups=pd.read_csv(OUT/'组合.csv')
summary_max_error=0.
cohort_reports={}

for ci,cfg in enumerate(plan['configurations']):
    ident=cfg['id']
    design=json.loads((CACHE/f'{ident}_设计.json').read_text(encoding='utf-8'))
    samplefile=CACHE/f'{ident}_抽样.npz'
    assert sha(samplefile)==plan['sampling_sha256'][samplefile.name]
    thresholdfile=CACHE/f'{ident}_阈值.npz'
    assert sha(thresholdfile)==freeze['files_sha256'][thresholdfile.name]
    with np.load(samplefile,allow_pickle=False) as z: sam=dict(z)
    with np.load(thresholdfile,allow_pickle=False) as z: th,anchors=z['thresholds'],z['anchors']
    with np.load(CACHE/f'{ident}_逐人.npz',allow_pickle=False) as z: rates,people=z['rates'],z['subjects']
    expected_full=2**len(cfg['cal_people'])-1
    assert len(th)==expected_full*(1+3*1000)
    assert np.isfinite(th).all() and np.all(np.diff(th,axis=3)<=0)
    assert np.isfinite(rates).all() and rates.min()>=0 and rates.max()<=1+1e-12
    assert np.all(np.diff(rates,axis=3)>=-1e-14)
    assert np.array_equal(people,cfg['test_people'])
    assert not set(cfg['cal_people']) & set(cfg['test_people'])
    total_blocks=plan['inventory'][ident]['total_blocks']
    minima=min(plan['inventory'][ident]['background_endpoints_per_person'].values())
    div=math.lcm(*range(1,len(cfg['cal_people'])+1))*4
    assert total_blocks==minima//50//div*div
    for sid in cfg['cal_people']:
        b=sam[f'blocks_{sid}']
        mask=(sam['base_subjects']==sid)&(sam['base_labels']==0)
        assert len(b)==int(mask.sum())
        # Strong inclusion check: over all equally likely block starts, every
        # background endpoint occurs exactly 50 times, all action points zero.
        occurrences=np.bincount(b.ravel(),minlength=len(mask))
        assert np.array_equal(occurrences,mask.astype(np.int64)*50)
        checks['background_endpoints_verified']+=int(mask.sum())
        rng=np.random.default_rng(np.random.SeedSequence([20261112,ci,sid]))
        assert np.array_equal(sam[f'draws_{sid}'],rng.integers(len(b),size=(1000,total_blocks),dtype=np.int32))
        rows=sam['base_rows'][b]
        assert np.all((np.diff(rows,axis=1)==10)|(np.diff(rows,axis=1)<=0))
    selected=[]
    for d in design:
        assert not set(d['group'])&set(cfg['test_people'])
        assert len(d['group'])==d['k']
        selected.append(d['start'])
    direct=np.zeros_like(rates[selected])
    for si,seed in enumerate(cfg['seeds']):
        p=pack(cfg,seed,'cal')
        base=np.flatnonzero((p['scenario']==0)&(p['direction']==0))
        assert np.array_equal(p['row'][base],sam['base_rows'])
        subject,y=p['subject'][base],p['y'][base]
        for dr in [0,1]:
            slots=[np.flatnonzero((p['scenario']==a)&(p['direction']==dr)) for a in range(4)]
            scores=np.stack([p['scores'][ix] for ix in slots],axis=1)
            for d in design:
                offsets=[0] if d['kind']=='full' else [0,999]
                for offset in offsets:
                    row=d['start']+offset
                    if d['kind']=='full':
                        ix=np.flatnonzero(np.isin(subject,d['group'])&(y==0))
                        weight=np.zeros(len(ix))
                        for sid in d['group']:
                            take=subject[ix]==sid
                            weight[take]=1/(d['k']*take.sum())
                    else:
                        nper=round(total_blocks*d['fraction'])//d['k']
                        ix=np.concatenate([sam[f'blocks_{sid}'][sam[f'draws_{sid}'][offset,:nper]].ravel() for sid in d['group']])
                        assert len(ix)==round(total_blocks*d['fraction'])*50
                        weight=np.full(len(ix),1/len(ix))
                    for m in range(3):
                        for column,anchor in enumerate([None,2,3]):
                            score=scores[ix,:,m].ravel() if anchor is None else scores[ix,anchor,m]
                            w=np.repeat(weight/4,4) if anchor is None else weight
                            actual=np.array([cdf_threshold(score,w,a) for a in targets])
                            np.testing.assert_allclose(th[row,si*2+dr,m,:,column],actual,rtol=0,atol=1e-12)
                            checks['calibration_values_reconstructed']+=4
            if cfg['split']==0:
                full=[d for d in design if d['kind']=='full' and d['k']==len(cfg['cal_people'])][0]
                previous=old[f'{cfg["dataset"]}_{seed}'][str(dr)]
                np.testing.assert_allclose(anchors[full['start']],previous['knots'],rtol=0,atol=1e-12)
                for m,mode in enumerate(plan['modes']):
                    for ti,a in enumerate(targets):
                        v=previous['policies'][mode][str(a)]
                        np.testing.assert_allclose(th[full['start'],si*2+dr,m,ti],[v['global'],v['anchors'][2],v['anchors'][3]],rtol=0,atol=1e-12)
        p=pack(cfg,seed,'test')
        for dr in [0,1]:
            for pi,sid in enumerate(people):
                ix=np.flatnonzero((p['scenario']==3)&(p['direction']==dr)&(p['subject']==sid))
                y,age=p['y'][ix],p['age'][ix]
                for ri,row in enumerate(selected):
                    for m in range(3):
                        score,action=p['scores'][ix,m],p['actions'][ix,m]
                        for ti,target in enumerate(targets):
                            t=th[row,si*2+dr,m,ti]
                            policies=[t[0],np.interp(age,anchors[row,2:],t[1:])]
                            for po,tau in enumerate(policies):
                                pred=np.where(score>tau,action,0)
                                direct[ri,m,po,ti,pi,0]+=np.mean(pred[y==0]>0)/6
                                direct[ri,m,po,ti,pi,1]+=np.mean(pred[y>0]==y[y>0])/6
                                checks['direct_evaluation_rates_reconstructed']+=2
        print('AUDIT',ident,seed,flush=True)
    np.testing.assert_allclose(rates[selected],direct,rtol=0,atol=1e-12)
    if cfg['split']==0:
        full=[d for d in design if d['kind']=='full' and d['k']==len(cfg['cal_people'])][0]
        for m,mode in enumerate(plan['modes']):
            for po,policy in enumerate(plan['policies']):
                for ti,target in enumerate(targets):
                    for pi,sid in enumerate(people):
                        row=old_rates[(old_rates.dataset==cfg['dataset'])&(old_rates.scenario.astype(str)=='1000')&(old_rates['mode']==mode)&(old_rates.policy==policy)&(np.abs(old_rates.target-target)<1e-12)&(old_rates.subject==sid)]
                        assert len(row)==1
                        np.testing.assert_allclose(rates[full['start'],m,po,ti,pi],row[['fpr','car']].to_numpy()[0],rtol=0,atol=1e-12)
                        checks['original_reference_rate_matches']+=2
    # Recalculate every published summary from participant-level replicate arrays.
    for frame,scope in [(summary,'size'),(groups,'group')]:
        rows=frame[(frame.dataset==cfg['dataset'])&(frame['split']==cfg['split'])]
        for _,row in rows.iterrows():
            chosen=[d for d in design if d['kind']==row['kind'] and d['k']==row.k and d['fraction']==row.fraction]
            if scope=='group':
                key=str(row.group)
                chosen=[d for d in chosen if ','.join(map(str,d['group']))==key]
            assert chosen
            v=np.concatenate([rates[d['start']:d['start']+d['count']] for d in chosen])
            m=plan['modes'].index(row['mode']); po=plan['policies'].index(row.policy)
            ti=int(np.argmin(np.abs(targets-row.target)))
            f=v[:,m,po,ti,:,0]; c=v[:,m,po,ti,:,1]; avg=f.mean(axis=1)
            lo,hi=np.percentile(avg,[2.5,97.5])
            vals=dict(mean_fpr=avg.mean(),conditional_low=lo,conditional_high=hi,
                min_fpr=avg.min(),max_fpr=avg.max(),mean_car=c.mean(),
                mean_signed_gap=avg.mean()-row.target,mean_absolute_mean_gap=np.abs(avg-row.target).mean(),
                mean_participant_absolute_gap=np.abs(f-row.target).mean(),mean_worst_participant_fpr=f.max(axis=1).mean())
            error=max(abs(float(row[k])-v) for k,v in vals.items())
            assert error<1e-12
            summary_max_error=max(summary_max_error,error)
    cohort_reports[ident]=dict(evaluation_people=people.tolist(),calibration_people=cfg['cal_people'],
        unique_calibration_subsets=expected_full,resample_rows=len(rates)-expected_full,
        background_budget_per_anchor=total_blocks*50)

report=dict(status='PASS_COMPUTATIONAL_AUDIT',utc=datetime.now(timezone.utc).isoformat(),
    scope='Recorded post-exposure sensitivity analysis only; no independent replication, causal shift attribution or population-level confidence guarantee.',
    checks=checks,summary_rows_checked=len(summary),group_summary_rows_checked=len(groups),
    summary_max_absolute_error=summary_max_error,cohorts=cohort_reports,
    all_original_background_endpoints_have_equal_expected_sampling_weight=True,
    models_and_evaluation_participants_fixed=True,original_reference_results_reproduced=True,
    chinese_manuscript_unchanged=True,
    plan_sha256=sha(OUT/'方案.json'),analysis_code_sha256=sha(OUT/'分析.py'),audit_code_sha256=sha(__file__),
    outputs_sha256={p.name:sha(p) for p in OUT.glob('*.csv')},
    detailed_output_sha256={p.name:sha(p) for p in CACHE.glob('*.npz')})
(OUT/'核验.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
done=json.loads((OUT/'完成.json').read_text(encoding='utf-8'))
done['status']='COMPLETE_AUDITED_CALIBRATION_SENSITIVITY'
done['audit_sha256']=sha(OUT/'核验.json')
(OUT/'完成.json').write_text(json.dumps(done,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if 'sha256' not in k},ensure_ascii=False,indent=2),flush=True)
