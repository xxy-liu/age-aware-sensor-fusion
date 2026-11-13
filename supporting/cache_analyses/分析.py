"""Prediction-change and calibration-transfer analyses from saved predictions.

The fixed protocol separates threshold calibration, evaluation and numerical
verification. Inputs are the existing model-output caches.
"""
from pathlib import Path
import sys, json, hashlib, platform
from datetime import datetime, timezone
import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SRC = ROOT / '03_实验记录'
CONFIG = {
    'REALDISP': {'folder':'收尾实验', 'seeds':[9101,9102,9103],
        'cal':'calibration_REALDISP_CNN_{seed}.npz', 'test':'evaluate_REALDISP_CNN_{seed}.npz',
        'names':['aware_native','aware_zero','blind_native'], 'k':34,
        'cal_people':[9,10,11], 'test_people':[12,13,14,15,16,17]},
    'MHEALTH': {'folder':'前瞻验证', 'seeds':[9301,9302,9303],
        'cal':'calibration_{seed}.npz', 'test':'test_{seed}.npz',
        'names':['aware20_native','aware20_zero','blind20_native'], 'k':13,
        'cal_people':[8,10], 'test_people':[2,4,7,9]},
}
SCENES = ['0','167','500','1000','333','750','jitter','step','burst']
TARGETS = [.05,.10,.15,.20]
MODES = ['aware','zero','blind']
METRICS = ['macro_f1','car','fpr','miss','wrong_action','accuracy']
BOOT_SEED = 20261109

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def read(name):
    return json.loads((OUT/name).read_text(encoding='utf-8'))

def source_paths():
    paths=[]
    for cfg in CONFIG.values():
        for seed in cfg['seeds']:
            for role in ('cal','test'): paths.append(SRC/cfg['folder']/'预测'/cfg[role].format(seed=seed))
        paths += [SRC/cfg['folder']/n for n in ('方案.json','阈值.json','核验.json')]
    paths += [SRC/'独立补强'/n for n in ('阈值.csv','结果.csv','类别排序.csv')]
    return paths

def freeze():
    assert not (OUT/'方案.json').exists(), 'The fixed plan already exists.'
    plan={
        'date':'2026-11-09', 'utc':datetime.now(timezone.utc).isoformat(),
        'authorization':'Prediction-change and calibration-transfer analyses using saved model outputs, with numerical verification.',
        'status':'FROZEN_POST_EXPOSURE_CACHE_ANALYSIS',
        'scope':'原始分组的20轮CNN；两个数据集；三种子；两方向；全部九场景。只读既有预测，不训练、不推断、不解码原始传感器记录、不读取额外正式测试。',
        'datasets':CONFIG, 'scenarios':SCENES, 'targets':TARGETS,
        'analysis_1':{
            'question':'阈值匹配后相近的净指标是否掩盖纠错和新增错误；接受动作与动作类别选择分别如何变化？',
            'prediction_pairs':'同一aware模型，reference为native且阈值0；baseline分别为zero且阈值0、zero且原四锚点FPR匹配阈值按实际年龄插值。匹配阈值仅由原校准缓存拟合。',
            'outputs':'逐人/种子/方向/场景的纠错、改错、错转错、预测不一致计数及比例；背景/动作分母分别保存；检测开关、两者均接受时的动作类别改变、未阈值化动作排名不一致。',
            'identity':'reference accuracy minus baseline accuracy=(corrected-harmed)/N。Macro-F1不可用该恒等式相加分解。',
            'crossed_readouts':'检测margin取native或zero，动作类别取native或zero；在四个校准目标下组合为4种读出，阈值使用各自检测分数的age-interpolated策略。交叉读出用于输出层面的诊断，不对应新的训练模型、部署干预或内部因果机制。',
        },
        'analysis_2':{
            'question':'目标15%的迁移现象在5/10/20%时是否仍存在？',
            'policies':['global','age_interpolated'], 'modes':MODES,
            'calibration':'每种子/模式/方向分别拟合。四锚点0/167/500/1000；背景内各人等权，global各锚点等权。阈值为满足经验加权FPR<=target的最小观测分数；严格margin>threshold；并列整组拒绝。',
            'transfer':'global应用一个阈值；age_interpolated对四锚点阈值按实际年龄线性插值，越界用端点；额外五场景不重新拟合。',
            'outputs':'全部目标、场景和模式的实际FPR、CAR、Macro-F1、漏检和错动作；逐人超目标比例、绝对偏差；不宣称人群风险保证。',
            'reuse':'原REALDISP四静态条件已有同类四目标结果，新代码核对原阈值与指标；新增覆盖是MHEALTH、动态迁移、global策略及逐样本变化。',
        },
        'aggregation':'先在每人内等权平均3种子和2方向，再人等权。不同目标/场景不混合为独立观察。',
        'uncertainty':{'method':'20000 paired participant bootstrap; conditional on fitted models and calibration allocations', 'seed':BOOT_SEED,'n_boot':20000,'inference':'描述性补充分析；不选择较好目标/场景，不做新增显著性筛选。'},
        'limits':'原评价对象已暴露。无新独立验证、无科学等效性检验、无真实网络部署、无通用新算法主张。',
        'stop':'分析范围限定为两组缓存分析及数值核验，模型、种子和数据集固定。',
        'software':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__},
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in source_paths()},
        'analysis_code_sha256':sha(__file__),
    }
    dump('方案.json',plan)
    print('FROZEN',len(plan['source_sha256']),flush=True)

def loadpack(cfg,seed,role):
    path=SRC/cfg['folder']/'预测'/cfg[role].format(seed=seed)
    with np.load(path,allow_pickle=False) as z:
        names=z['components'].tolist(); col=[names.index(n) for n in cfg['names']]
        p={k:z[k] for k in ('subject','scenario','direction','age','y','row','old')}
        p['margin']=z['margin'][:,col]; p['action']=z['action'][:,col]
        p['scenarios']=z['scenarios'].tolist()
    assert np.isfinite(p['margin']).all()
    assert p['margin'].shape == p['action'].shape
    assert sorted(np.unique(p['subject']).tolist())==sorted(cfg['cal_people' if role=='cal' else 'test_people'])
    assert p['scenarios']==SCENES[:4] if role=='cal' else p['scenarios']==SCENES
    return p

def threshold(score,weight,target):
    keep=weight>0;s=score[keep];w=weight[keep]
    order=np.argsort(s,kind='stable');s=s[order];w=w[order]
    values,first=np.unique(s,return_index=True);mass=np.add.reduceat(w,first)
    tail=np.r_[np.cumsum(mass[:0:-1])[::-1],0.]
    j=np.flatnonzero(tail<=target+1e-12)[0]
    return float(values[j])

def calibrate():
    plan=read('方案.json');assert sha(__file__)==plan['analysis_code_sha256']
    out={}
    for dataset,cfg in CONFIG.items():
        for seed in cfg['seeds']:
            p=loadpack(cfg,seed,'cal'); result={}
            for dr in (0,1):
                sel=p['direction']==dr; subjects=p['subject'][sel];sc=p['scenario'][sel]
                y=p['y'][sel];m=p['margin'][sel];age=p['age'][sel]
                w=np.zeros(len(y));wa=np.zeros(len(y))
                for sid in cfg['cal_people']:
                    for a in range(4):
                        q=(subjects==sid)&(sc==a);b=q&(y==0)
                        assert q.any() and b.any()
                        w[b]=1/(len(cfg['cal_people'])*4*b.sum())
                        wa[q]=1/(len(cfg['cal_people'])*4*q.sum())
                knots=[float((wa[sc==a]*4*age[sc==a]).sum()) for a in range(4)]
                assert np.all(np.diff(knots)>0)
                match=[]
                for a in range(4):
                    q=sc==a;ww=w[q]*4;target=float(ww[m[q,0]>0].sum())
                    match.append(threshold(m[q,1],ww,target))
                entry={'knots':knots,'zero_match':match,'policies':{}}
                for j,mode in enumerate(MODES):
                    entry['policies'][mode]={}
                    for target in TARGETS:
                        entry['policies'][mode][str(target)]={
                            'global':threshold(m[:,j],w,target),
                            'anchors':[threshold(m[sc==a,j],w[sc==a]*4,target) for a in range(4)]}
                result[str(dr)]=entry
            out[f'{dataset}_{seed}']=result
            print('CALIBRATED',dataset,seed,flush=True)
    dump('阈值.json',out)
    dump('阈值冻结.json',{'utc':datetime.now(timezone.utc).isoformat(),'plan_sha256':sha(OUT/'方案.json'),'threshold_sha256':sha(OUT/'阈值.json'),'analysis_code_sha256':sha(__file__),'test_cache_numeric_access_in_this_run':False})

def metrics(y,p,k):
    cm=np.bincount(y*k+p,minlength=k*k).reshape(k,k);den=cm.sum(0)+cm.sum(1)
    a=y>0;b=~a
    assert a.any() and b.any()
    return {'n':int(len(y)),'n_action':int(a.sum()),'n_background':int(b.sum()),
        'macro_f1':float(np.divide(2*cm.diagonal(),den,out=np.zeros(k),where=den>0).mean()),
        'car':float(np.mean(p[a]==y[a])),'fpr':float(np.mean(p[b]>0)),
        'miss':float(np.mean(p[a]==0)),'wrong_action':float(np.mean((p[a]>0)&(p[a]!=y[a]))),
        'accuracy':float(np.mean(p==y))}

def changes(y,ref,base,c_ref,c_base):
    good=ref==y;old=base==y;different=ref!=base;a=y>0;b=~a
    masks={'corrected':good&~old,'harmed':~good&old,'wrong_to_wrong':different&~good&~old,
           'disagreement':different,'gate_change':(ref>0)!=(base>0),
           'reported_action_change':(ref>0)&(base>0)&different,
           'action_rank_disagreement':c_ref!=c_base}
    d={'n':int(len(y)),'n_action':int(a.sum()),'n_background':int(b.sum())}
    for name,mask in masks.items():
        d[name+'_n']=int(mask.sum());d[name]=float(mask.mean())
        d[name+'_action']=float(mask[a].mean());d[name+'_background']=float(mask[b].mean())
    d['accuracy_delta']=float(good.mean()-old.mean())
    assert abs(d['accuracy_delta']-(d['corrected']-d['harmed']))<1e-14
    assert d['disagreement_n']==d['corrected_n']+d['harmed_n']+d['wrong_to_wrong_n']
    assert d['disagreement_n']==d['gate_change_n']+d['reported_action_change_n']
    return d

def summarize(frame,keys,values,name):
    person=frame.groupby(keys+['subject'],sort=True,dropna=False)[values].mean().reset_index()
    person.to_csv(OUT/(name+'逐人.csv'),index=False,encoding='utf-8-sig',float_format='%.17g')
    rows=[]
    for key,g in person.groupby(keys,sort=True,dropna=False):
        if not isinstance(key,tuple):key=(key,)
        g=g.sort_values('subject');v=g[values].to_numpy();n=len(g)
        ix=np.random.default_rng(BOOT_SEED).integers(0,n,(20000,n))
        boot=v[ix].mean(1);low,high=np.quantile(boot,[.025,.975],axis=0)
        for j,metric in enumerate(values):
            rows.append(dict(zip(keys,key),metric=metric,mean=float(v[:,j].mean()),low=float(low[j]),high=float(high[j]),n_people=n,minimum=float(v[:,j].min()),maximum=float(v[:,j].max())))
    pd.DataFrame(rows).to_csv(OUT/(name+'汇总.csv'),index=False,encoding='utf-8-sig',float_format='%.17g')

def evaluate():
    plan=read('方案.json');freeze_record=read('阈值冻结.json')
    assert sha(__file__)==plan['analysis_code_sha256']==freeze_record['analysis_code_sha256']
    assert sha(OUT/'阈值.json')==freeze_record['threshold_sha256']
    policies=read('阈值.json');flip_rows=[];rate_rows=[];cross_rows=[]
    for dataset,cfg in CONFIG.items():
        for seed in cfg['seeds']:
            p=loadpack(cfg,seed,'test')
            for dr in (0,1):
                pol=policies[f'{dataset}_{seed}'][str(dr)]
                for si,scene in enumerate(SCENES):
                    for sid in cfg['test_people']:
                        take=(p['direction']==dr)&(p['scenario']==si)&(p['subject']==sid)
                        y=p['y'][take];m=p['margin'][take];c=p['action'][take];age=p['age'][take]
                        meta={'dataset':dataset,'seed':seed,'direction':dr,'scenario':scene,'subject':sid}
                        native=np.where(m[:,0]>0,c[:,0],0)
                        for comparison,thr in [('raw',0.),('matched',np.interp(age,pol['knots'],pol['zero_match']))]:
                            baseline=np.where(m[:,1]>thr,c[:,1],0)
                            flip_rows.append(dict(**meta,comparison=comparison,**changes(y,native,baseline,c[:,0],c[:,1])))
                        for target in TARGETS:
                            local_thresholds=[]
                            for j,mode in enumerate(MODES):
                                fit=pol['policies'][mode][str(target)]
                                interp=np.interp(age,pol['knots'],fit['anchors']);local_thresholds.append(interp)
                                for policy,thr in [('global',fit['global']),('age_interpolated',interp)]:
                                    pred=np.where(m[:,j]>thr,c[:,j],0)
                                    rate_rows.append(dict(**meta,target=target,mode=mode,policy=policy,**metrics(y,pred,cfg['k'])))
                            for detector in (0,1):
                                gate=m[:,detector]>local_thresholds[detector]
                                for ranking in (0,1):
                                    pred=np.where(gate,c[:,ranking],0)
                                    cross_rows.append(dict(**meta,target=target,detector=MODES[detector],ranking=MODES[ranking],**metrics(y,pred,cfg['k'])))
            print('EVALUATED',dataset,seed,flush=True)
    for name,rows in [('预测变化',flip_rows),('校准迁移',rate_rows),('交叉读出',cross_rows)]:
        pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig',float_format='%.17g')
    f=pd.DataFrame(flip_rows)
    vals=[c for c in f if c not in ['dataset','seed','direction','scenario','subject','comparison','n','n_action','n_background'] and not c.endswith('_n')]
    summarize(f,['dataset','scenario','comparison'],vals,'预测变化')
    summarize(pd.DataFrame(rate_rows),['dataset','scenario','target','mode','policy'],METRICS,'校准迁移')
    summarize(pd.DataFrame(cross_rows),['dataset','scenario','target','detector','ranking'],METRICS,'交叉读出')
    files=list(OUT.glob('*.csv'))+[OUT/'阈值.json',OUT/'阈值冻结.json']
    dump('完成.json',{'utc':datetime.now(timezone.utc).isoformat(),'status':'COMPUTED_PENDING_AUDIT',
        'counts':{'prediction_changes':len(flip_rows),'calibration_metrics':len(rate_rows),'crossed_readouts':len(cross_rows)},
        'outputs_sha256':{p.name:sha(p) for p in files}})
    print('COMPUTATION_COMPLETE',len(flip_rows),len(rate_rows),len(cross_rows),flush=True)

if __name__=='__main__':
    {'freeze':freeze,'calibrate':calibrate,'evaluate':evaluate}[sys.argv[1]]()
