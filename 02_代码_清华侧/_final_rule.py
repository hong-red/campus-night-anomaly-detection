import os as _os, sys as _sys
_cur = _os.path.dirname(_os.path.abspath(__file__))
for _i in range(6):
    if _os.path.exists(_os.path.join(_cur, "_paths.py")):
        _sys.path.insert(0, _cur)
        break
    _cur = _os.path.dirname(_cur)
import _paths as _PKGP
import sys, json, numpy as np
sys.path.insert(0, _PKGP.CODE_TSINGHUA); sys.path.insert(0, _PKGP.TCN_DIR)
from sklearn import metrics
import tcn_branch as T, geolife_pipeline as gp
from compare_literature import (inject_dwell, inject_geofence, inject_gather,
                                ref_frame_scores, max_stationary_run)

DT=2.0
walks = gp.scan_walking_trajectories(T.DEFAULT_DATA, max_files=2000, min_dur=900)
clean=[]
for w in walks:
    lat,lon,tg = gp.resample(w[3],w[4],w[5],DT)
    sp,_,_ = gp.trajectory_features(lat,lon,DT)
    if max_stationary_run(sp,DT)<=120: clean.append(w)
print('clean=', len(clean))
evs = {"dwell":inject_dwell,"geofence":inject_geofence,"gather":inject_gather}
out={}
for key, inj in evs.items():
    pools={d:([],[]) for d in ["rules","spot","ijgi","applsci"]}
    for w in clean:
        lat,lon,tg = gp.resample(w[3],w[4],w[5],DT); n=len(lat)
        if n<int(700/DT): continue
        lat,lon,others,zone,gt = inj(lat,lon,n)
        x,y = T.latlon_to_local(lat,lon)
        sc = ref_frame_scores(lat,lon,DT,others,zone,x,y)
        for d,s in sc.items():
            pools[d][0].append(gt); pools[d][1].append(s)   # 不掩码
    out[key]={d: round(metrics.roc_auc_score(np.concatenate(pools[d][0]), np.concatenate(pools[d][1])),3) for d in pools}
    print(key, out[key])
json.dump(out, open('results/lit_compare/_rule_unmasked.json','w'), indent=2)
