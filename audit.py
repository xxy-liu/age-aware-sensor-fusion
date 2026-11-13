"""Independently check Table II counts, threshold minimality and allocations."""
from pathlib import Path
import csv
import json
import platform

import numpy as np

BASE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    report_path = BASE/'outputs/independent_audit.json'
    if report_path.exists():
        report_path.unlink()
    config = read(BASE/'config/evaluation.json')
    policies = read(BASE/'config/thresholds.json')
    recorded = csv_rows(BASE/'outputs/table2_runs.csv')
    lookup = {(r['dataset'],int(r['seed']),int(r['direction']),int(r['subject']),r['readout']):r for r in recorded}
    allocations = csv_rows(BASE/'config/participants.csv')
    groups_checked = 0
    for g in config['groups']:
        role_sets = []
        for role, key in [('train','train'),('calibration','calibration'),('test','evaluation')]:
            a = sorted((r for r in allocations if r['dataset']==g['dataset'] and r['partition']==g['partition'] and r['role']==role), key=lambda r:int(r['order']))
            ids = [int(r['participant']) for r in a]
            if ids != g[key] or len(ids) != len(set(ids)):
                raise ValueError('Allocation order/IDs differ')
            role_sets.append(set(ids))
        if any(role_sets[i]&role_sets[j] for i in range(3) for j in range(i)):
            raise ValueError('Roles overlap within a partition')
        if set.union(*role_sets) != set(range(1,18 if g['dataset']=='REALDISP' else 11)):
            raise ValueError('Incomplete partition')
        groups_checked += 1
    threshold_checks, numeric_rows, max_error = [],0,0.
    for g in config['groups']:
        if g['partition']!='O' or g['architecture']!='CNN':
            continue
        native = 'aware_native' if g['dataset']=='REALDISP' else 'aware20_native'
        zero = 'aware_zero' if g['dataset']=='REALDISP' else 'aware20_zero'
        for run in g['runs']:
            with np.load(BASE/run['calibration'],allow_pickle=False) as z:
                names=z['components'].tolist(); si=z['scenarios'].tolist().index('1000')
                subject,y,scenario,direction,scores=z['subject'],z['y'],z['scenario'],z['direction'],z['margin']
                for dr in [0,1]:
                    index=(scenario==si)&(direction==dr)&(y==0)
                    ids=subject[index]; m=scores[index]
                    counts={sid:np.count_nonzero(ids==sid) for sid in g['calibration']}
                    w=np.array([1/(len(g['calibration'])*counts[sid]) for sid in ids])
                    target=float(sum(np.mean(m[ids==sid,names.index(native)]>0) for sid in g['calibration'])/len(g['calibration']))
                    s=m[:,names.index(zero)]
                    t=policies[run['id']][str(dr)]['components'][zero]['local_match'][si]
                    feasible=float(w[s>t].sum()) <= target+1e-12
                    unique=np.unique(s)
                    at=np.searchsorted(unique,t)
                    if at==len(unique) or unique[at]!=t:
                        raise ValueError('Threshold is not an observed score')
                    predecessor=None if at==0 else float(w[s>unique[at-1]].sum())
                    minimal=predecessor is None or predecessor>target+1e-12
                    if not feasible or not minimal:
                        raise ValueError('Threshold is not the least conservative feasible observed score')
                    threshold_checks.append(dict(run=run['id'],direction=dr,feasible=feasible,minimal=minimal))
            with np.load(BASE/run['evaluation'],allow_pickle=False) as z:
                names=z['components'].tolist(); si=z['scenarios'].tolist().index('1000')
                subject,y,scenario,direction,scores,actions=z['subject'],z['y'],z['scenario'],z['direction'],z['margin'],z['action']
                for dr in [0,1]:
                    t=policies[run['id']][str(dr)]['components'][zero]['local_match'][si]
                    for sid in g['evaluation']:
                        q=(scenario==si)&(direction==dr)&(subject==sid)
                        labels=y[q]; true_counts=np.bincount(labels,minlength=g['classes'])
                        for label,name,threshold in [('Native',native,0.),('Zero',zero,0.),('Zero+match',zero,t)]:
                            j=names.index(name)
                            pred=actions[q,j].copy()
                            pred[scores[q,j]<=threshold]=0
                            pred_counts=np.bincount(pred,minlength=g['classes'])
                            tp=np.bincount(labels[pred==labels],minlength=g['classes'])
                            den=true_counts+pred_counts
                            class_f1=[0. if den[k]==0 else 2*int(tp[k])/int(den[k]) for k in range(g['classes'])]
                            active=labels!=0; bg=~active
                            values=dict(macro_f1=sum(class_f1)/g['classes'],
                                        car=np.count_nonzero((pred==labels)&active)/active.sum(),
                                        fpr=np.count_nonzero((pred!=0)&bg)/bg.sum())
                            reference=lookup[(g['dataset'],run['seed'],dr,sid,label)]
                            if len(labels)!=int(reference['n']):
                                raise ValueError('Query denominator mismatch')
                            error=max(abs(float(value)-float(reference[key])) for key,value in values.items())
                            max_error=max(max_error,error)
                            if error>2e-12:
                                raise ValueError('Independent count reconstruction mismatch')
                            numeric_rows+=1
    if numeric_rows!=len(recorded):
        raise ValueError('Unverified Table II run rows')
    result=dict(status='PASS',python=platform.python_version(),numpy=np.__version__,
                allocations_checked=groups_checked,metric_rows_checked=numeric_rows,
                maximum_absolute_error=max_error,threshold_checks=threshold_checks,
                scope='Independent marginal-count metric calculation and observed-score threshold minimality; no new inference.')
    report_path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('PASS independent audit:',numeric_rows,'run rows;',len(threshold_checks),'minimal thresholds')


if __name__=='__main__':
    main()
