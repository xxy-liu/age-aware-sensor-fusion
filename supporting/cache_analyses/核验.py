"""Separate numerical audit of the frozen cache analysis; no training imports."""
from pathlib import Path
import json, hashlib, sys
import numpy as np
import pandas as pd
sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];SRC=ROOT/'03_实验记录'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
plan=read(OUT/'方案.json');cal=read(OUT/'阈值.json');done=read(OUT/'完成.json')
assert sha(OUT/'分析.py')==plan['analysis_code_sha256']
for rel,h in plan['source_sha256'].items():assert sha(ROOT/rel)==h,rel
for rel,h in done['outputs_sha256'].items():assert sha(OUT/rel)==h,rel
tables={n:pd.read_csv(OUT/(n+'.csv'),float_precision='round_trip',dtype={'scenario':str}) for n in ['预测变化','校准迁移','交叉读出']}
rate_keys=['dataset','seed','direction','scenario','subject','target','mode','policy']
cross_keys=['dataset','seed','direction','scenario','subject','target','detector','ranking']
flip_keys=['dataset','seed','direction','scenario','subject','comparison']
saved={n:tables[n].set_index(keys) for n,keys in [('校准迁移',rate_keys),('交叉读出',cross_keys),('预测变化',flip_keys)]}
for df in saved.values():assert df.index.is_unique
metric_names=['macro_f1','car','fpr','miss','wrong_action','accuracy']
max_error=0.;threshold_count=0;metric_count=0;flip_count=0;old_thresholds=0
historical_threshold_max_difference=0.;historical_threshold_nonidentical=0
cache_replay_checks=[]

def check_metric(y,pred,k,row):
 global max_error,metric_count
 active=y!=0;bg=~active
 true_count=np.bincount(y,minlength=k);pred_count=np.bincount(pred,minlength=k)
 tp=np.bincount(y[y==pred],minlength=k);den=true_count+pred_count
 f1=np.zeros(k);nonzero=den!=0;f1[nonzero]=2.*tp[nonzero]/den[nonzero]
 expected=[f1.mean(),np.count_nonzero((pred==y)&active)/active.sum(),
  np.count_nonzero((pred!=0)&bg)/bg.sum(),np.count_nonzero((pred==0)&active)/active.sum(),
  np.count_nonzero((pred!=0)&(pred!=y)&active)/active.sum(),np.count_nonzero(pred==y)/len(y)]
 err=np.max(np.abs(np.array(expected)-row[metric_names].to_numpy(dtype=float)))
 max_error=max(max_error,float(err));assert err<2e-12,err
 assert row['n']==len(y) and row['n_action']==active.sum() and row['n_background']==bg.sum()
 assert abs(expected[1]+expected[3]+expected[4]-1)<1e-12
 metric_count+=1

def quantile_threshold(s,w,t):
 q=w>0;s=s[q];w=w[q];order=np.argsort(s,kind='stable')
 cumulative=np.cumsum(w[order]);i=np.searchsorted(cumulative,1-t-1e-12,side='left')
 return float(s[order[min(i,len(s)-1)]])

historical=pd.read_csv(SRC/'独立补强'/'阈值.csv',float_precision='round_trip')
for dataset,cfg in plan['datasets'].items():
 hist=read(SRC/cfg['folder']/'阈值.json')
 for seed in cfg['seeds']:
  with np.load(SRC/cfg['folder']/'预测'/cfg['cal'].format(seed=seed),allow_pickle=False) as z:
   columns=[z['components'].tolist().index(n) for n in cfg['names']]
   sy=z['subject'];scene=z['scenario'];direction=z['direction'];y=z['y'];ages=z['age'];m=z['margin'][:,columns]
   for dr in (0,1):
    pol=cal[f'{dataset}_{seed}'][str(dr)];hkey=f'REALDISP_CNN_{seed}' if dataset=='REALDISP' else str(seed)
    old=hist[hkey][str(dr)]['components']
    mask=direction==dr;idx=np.flatnonzero(mask);sub=sy[idx];sc=scene[idx];yy=y[idx];mm=m[idx];aa=ages[idx]
    weights=np.zeros(len(idx));wa=np.zeros(len(idx))
    for person in cfg['cal_people']:
     for si in range(4):
      take=(sub==person)&(sc==si);b=take&(yy==0)
      weights[b]=1./len(cfg['cal_people'])/4/b.sum();wa[take]=1./len(cfg['cal_people'])/4/take.sum()
    knots=[float(np.dot(wa[sc==s]*4,aa[sc==s])) for s in range(4)]
    assert np.allclose(knots,pol['knots'],atol=1e-14,rtol=0)
    assert np.allclose(pol['zero_match'],old[cfg['names'][1]]['local_match'],atol=0,rtol=0)
    for j,mode in enumerate(['aware','zero','blind']):
     for target in plan['targets']:
      fit=pol['policies'][mode][str(target)]
      assert fit['global']==quantile_threshold(mm[:,j],weights,target)
      assert weights[mm[:,j]>fit['global']].sum()<=target+1e-12
      threshold_count+=1
      for si,t in enumerate(fit['anchors']):
       use=sc==si;w=weights[use]*4;s=mm[use,j]
       assert t==quantile_threshold(s,w,target)
       assert w[s>t].sum()<=target+1e-12;threshold_count+=1
       if dataset=='REALDISP' and j<2:
        q=historical[(historical.seed==seed)&(historical.arm=='joint')&(historical.delay==int(plan['scenarios'][si]))&(historical.direction==dr)&(historical.policy==f'fpr{int(target*100):02d}')&(historical.detector==(1 if j==0 else 0))]
        assert len(q)==1
        gap=abs(float(q.iloc[0].threshold)-t)
        historical_threshold_max_difference=max(historical_threshold_max_difference,gap)
        historical_threshold_nonidentical+=int(gap>0)
        assert gap<.001, ('Historical cache replay tolerance',gap)
        old_thresholds+=1
      if target==.15:
       assert fit['global']==old[cfg['names'][j]]['global15']
       assert fit['anchors']==old[cfg['names'][j]]['local15']
   print('AUDIT_CALIBRATION',dataset,seed,flush=True)
  if dataset=='REALDISP':
   for role in ('cal','test'):
    oldrole='calibration' if role=='cal' else 'test'
    oldpath=SRC/'独立补强'/'预测'/f'{oldrole}_{seed}_joint.npz'
    newpath=SRC/cfg['folder']/'预测'/cfg[role].format(seed=seed)
    with np.load(oldpath,allow_pickle=False) as oldp, np.load(newpath,allow_pickle=False) as newp:
     newtake=newp['scenario']<4
     oldorder=np.lexsort((oldp['row'],oldp['subject'],oldp['direction'],oldp['delay']))
     newdelay=np.array([0,167,500,1000])[newp['scenario'][newtake]]
     neworder=np.lexsort((newp['row'][newtake],newp['subject'][newtake],newp['direction'][newtake],newdelay))
     for field in ('row','subject','direction','y'):
      assert np.array_equal(oldp[field][oldorder],newp[field][newtake][neworder])
     col=[newp['components'].tolist().index(x) for x in ['aware_zero','aware_native']]
     om=oldp['margin'][oldorder];nm=newp['margin'][newtake][neworder][:,col]
     oc=oldp['action'][oldorder];nc=newp['action'][newtake][neworder][:,col]
     gap=float(np.max(np.abs(om-nm)));assert gap<.001
     rank_diff=int(np.count_nonzero(oc!=nc))
     pred_diff=int(np.count_nonzero(np.where(om>0,oc,0)!=np.where(nm>0,nc,0)))
     assert rank_diff==0 and pred_diff==0
     cache_replay_checks.append({'seed':seed,'role':role,'max_margin_difference':gap,'action_differences':rank_diff,'default_prediction_differences':pred_diff,'historical_file_sha256':sha(oldpath)})
  with np.load(SRC/cfg['folder']/'预测'/cfg['test'].format(seed=seed),allow_pickle=False) as z:
   columns=[z['components'].tolist().index(n) for n in cfg['names']]
   sy=z['subject'];scene=z['scenario'];direction=z['direction'];y=z['y'];ages=z['age'];m=z['margin'][:,columns];c=z['action'][:,columns]
   assert set(np.unique(sy)).isdisjoint(cfg['cal_people'])
   for dr in (0,1):
    pol=cal[f'{dataset}_{seed}'][str(dr)]
    for si,sname in enumerate(plan['scenarios']):
     for sid in cfg['test_people']:
      ix=(direction==dr)&(scene==si)&(sy==sid);yy=y[ix];mm=m[ix];cc=c[ix];a=ages[ix]
      meta=(dataset,seed,dr,sname,sid);ref=cc[:,0].copy();ref[mm[:,0]<=0]=0
      for name,thr in [('raw',0.),('matched',np.interp(a,pol['knots'],pol['zero_match']))]:
       base=cc[:,1].copy();base[mm[:,1]<=thr]=0;row=saved['预测变化'].loc[meta+(name,)]
       statuses={'corrected':(ref==yy)&(base!=yy),'harmed':(ref!=yy)&(base==yy),
        'wrong_to_wrong':(ref!=base)&(ref!=yy)&(base!=yy),'disagreement':ref!=base,
        'gate_change':(ref!=0)!=(base!=0),'reported_action_change':(ref!=0)&(base!=0)&(ref!=base),
        'action_rank_disagreement':cc[:,0]!=cc[:,1]}
       for key,value in statuses.items():
        assert row[key+'_n']==value.sum()
        for suffix,subset in [('',np.ones(len(yy),bool)),('_action',yy>0),('_background',yy==0)]:
         assert abs(row[key+suffix]-np.count_nonzero(value&subset)/subset.sum())<1e-14
       assert abs(row.accuracy_delta-(row.corrected-row.harmed))<1e-14
       flip_count+=1
      for target in plan['targets']:
       for j,mode in enumerate(['aware','zero','blind']):
        fit=pol['policies'][mode][str(target)]
        for policy,t in [('global',fit['global']),('age_interpolated',np.interp(a,pol['knots'],fit['anchors']))]:
         pred=cc[:,j].copy();pred[mm[:,j]<=t]=0
         check_metric(yy,pred,cfg['k'],saved['校准迁移'].loc[meta+(target,mode,policy)])
       for det in (0,1):
        mode=['aware','zero'][det];thr=np.interp(a,pol['knots'],pol['policies'][mode][str(target)]['anchors'])
        for rank in (0,1):
         pred=cc[:,rank].copy();pred[mm[:,det]<=thr]=0
         check_metric(yy,pred,cfg['k'],saved['交叉读出'].loc[meta+(target,mode,['aware','zero'][rank])])
   print('AUDIT_PREDICTIONS',dataset,seed,flush=True)

aggregation_error=0.;ci_error=0.;summary_rows=0
for name,keys in [('预测变化',['dataset','scenario','comparison']),('校准迁移',['dataset','scenario','target','mode','policy']),('交叉读出',['dataset','scenario','target','detector','ranking'])]:
 raw=tables[name]
 people=pd.read_csv(OUT/(name+'逐人.csv'),float_precision='round_trip',dtype={'scenario':str})
 summary=pd.read_csv(OUT/(name+'汇总.csv'),float_precision='round_trip',dtype={'scenario':str})
 values=[c for c in people if c not in keys+['subject']]
 expected=raw.groupby(keys+['subject'],sort=True)[values].mean().sort_index()
 actual=people.set_index(keys+['subject']).sort_index()
 err=np.max(np.abs(expected.to_numpy()-actual.to_numpy()));aggregation_error=max(aggregation_error,float(err));assert err<1e-14
 for key,g in people.groupby(keys,sort=True):
  if not isinstance(key,tuple):key=(key,)
  g=g.sort_values('subject');v=g[values].to_numpy();n=len(g)
  sample=np.random.default_rng(20261109).integers(0,n,(20000,n));counts=np.zeros((20000,n))
  for j in range(n):counts[:,j]=(sample==j).sum(1)
  boot=counts@v/n;lo,hi=np.quantile(boot,[.025,.975],axis=0)
  part=summary
  for k,x in zip(keys,key):part=part[part[k]==x]
  part=part.set_index('metric').loc[values]
  err=np.max(np.abs(part[['mean','low','high']].to_numpy()-np.column_stack([v.mean(0),lo,hi])))
  ci_error=max(ci_error,float(err));assert err<1e-12
  summary_rows+=len(part)
report={'status':'PASS_COMPUTATIONAL_AUDIT','scope':'Numerical consistency checks using a separate implementation; no independent scientific replication.',
 'source_files_unchanged':len(plan['source_sha256']),'new_model_training':False,'new_model_inference':False,'raw_sensor_access':False,
 'new_threshold_checks':threshold_count,'historical_REALDISP_threshold_matches':old_thresholds,
 'historical_threshold_max_difference':historical_threshold_max_difference,'historical_threshold_nonidentical':historical_threshold_nonidentical,'historical_cache_replay':cache_replay_checks,
 'prediction_change_rows_checked':flip_count,'metric_rows_checked':metric_count,'summary_rows_checked':summary_rows,
 'max_metric_error':max_error,'max_aggregation_error':aggregation_error,'max_bootstrap_error':ci_error,
 'target_grid_complete':plan['targets'],'audit_script_sha256':sha(__file__),'outputs_sha256':done['outputs_sha256']}
(OUT/'核验.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='outputs_sha256'},ensure_ascii=False),flush=True)
