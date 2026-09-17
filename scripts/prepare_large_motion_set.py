#!/usr/bin/env python3
"""Restartable, train-split-only KIT retargeting and flat-ground admission.

Each source has a separate log/result. No source data or existing three-motion
outputs are changed. Manifests are atomically replaced, never directory-globbed.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import concurrent.futures
import contextlib
import hashlib
import json
from pathlib import Path
import time
import itertools
import numpy as np
import mujoco
from retarget_amass import ROOT, MODEL, retarget, MOTIONS

OUT = ROOT / 'outputs/retargeted/human39_large_v1'
REPORT = ROOT / 'reports/large_motion_set_v1'
LIMITS = dict(penetration_m=.005, ground_gap_m=.005, joint_speed_rad_s=10., slip_p95_m_s=.4)

def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)

def category(motion):
    name = motion.lower()
    if 'counterclockwise' in name or 'leftturn' in name or 'turn_left' in name:
        return 1
    if 'clockwise' in name or 'rightturn' in name or 'turn_right' in name:
        return 2
    return 0

def quality(path):
    with np.load(path) as z:
        q, v = z['qpos'], z['qvel']
        finite = all(np.isfinite(z[k]).all() for k in ('qpos','qvel','xpos','xquat'))
        if not finite:
            return {'accepted': False, 'reason': 'nonfinite'}, []
        m = mujoco.MjModel.from_xml_path(str(MODEL)); d = mujoco.MjData(m)
        geoms = [i for i in range(m.ngeom) if m.geom_contype[i] and
                 any(s in m.body(m.geom_bodyid[i]).name for s in ('Foot','Forefoot'))]
        corners = np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
        heights, positions = [], []
        for frame in q:
            d.qpos[:] = frame; mujoco.mj_forward(m,d)
            heights.append([np.min((d.geom_xpos[g] + (corners*m.geom_size[g]) @ d.geom_xmat[g].reshape(3,3).T)[:,2]) for g in geoms])
            positions.append(d.geom_xpos[geoms].copy())
        heights = np.array(heights)
        speeds = np.linalg.norm(np.gradient(np.array(positions), .02, axis=0)[:,:,:2], axis=-1)
        stance = heights < .008
        excess = np.maximum(np.maximum(m.jnt_range[1:,0]-q[:,7:], q[:,7:]-m.jnt_range[1:,1]),0).max()
        report = dict(finite=finite, joint_limit_excess_rad=float(excess),
                      penetration_m=float(max(0,-heights.min())), ground_gap_m=float(heights.min(axis=1).max()),
                      joint_speed_rad_s=float(abs(v[:,6:]).max()),
                      slip_p95_m_s=float(np.percentile(speeds[stance],95)) if stance.any() else 1e6,
                      minimum_root_height_m=float(q[:,2].min()))
        moving = np.flatnonzero(np.linalg.norm(v[:,:2],axis=1) > .12)
        # Preserve the original gate. Ground correction is unsuitable for aerial
        # motion; these sources were excluded before IK, not flattened and used.
        checks = {k: report[k] <= limit for k,limit in LIMITS.items()}
        checks.update(joint_limits=bool(excess<1e-6), upright=report['minimum_root_height_m']>.55,
                      moving=len(moving)>0)
        report['checks'] = checks; report['accepted'] = all(checks.values())
        clips = []
        if report['accepted']:
            lo, hi = max(0,int(moving[0])-25), min(len(q),int(moving[-1])+26)
            # Split only after source-level train/test assignment. At most 6 s,
            # with .5 s overlap, so full-clip evaluation fits the 8 s episode.
            start = lo
            while hi-start >= 50:
                end = min(start+300, hi)
                clips.append([start,end])
                if end == hi: break
                start = end-25
        return report, clips

def worker(motion, retry_errors_only=False):
    key = motion.replace('/','__')
    result_path = REPORT / (key+'.json')
    src = Path(json.loads((ROOT/'configs/paths.json').read_text())['raw_kit']).parent / (motion+'.npz')
    fingerprints = {'source':hashlib.sha256(src.read_bytes()).hexdigest(),
                    'model':hashlib.sha256(MODEL.read_bytes()).hexdigest(),
                    'retarget':hashlib.sha256((ROOT/'scripts/retarget_amass.py').read_bytes()).hexdigest(),
                    'gate':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if result_path.exists():
        old=json.loads(result_path.read_text())
        # Explicit retry mode preserves successful outputs and their ORIGINAL
        # implementation fingerprints; it does not relabel them as regenerated.
        if retry_errors_only and old.get('status')=='done' and (OUT/(key+'.npz')).exists():
            assert old['fingerprints']['source']==fingerprints['source']
            assert old['fingerprints']['model']==fingerprints['model']
            return old
        if old.get('fingerprints')==fingerprints and old.get('status')=='done' and (OUT/(key+'.npz')).exists():
            return old
    result={'motion':motion,'name':key,'category':category(motion),'fingerprints':fingerprints}
    try:
        with (REPORT/(key+'.log')).open('w') as log, contextlib.redirect_stdout(log):
            retarget(key,motion,50,OUT)
        gate, clips = quality(OUT/(key+'.npz'))
        result.update(status='done',quality=gate,clips=clips,path=str((OUT/(key+'.npz')).relative_to(ROOT)))
    except Exception as exc:
        result.update(status='error',error=repr(exc))
    write_json(result_path,result)
    return result

def publish(results, excluded, total, complete=False):
    accepted=[]
    # Keep original three clips first, preserving regression evaluation IDs.
    trims={'walk':[132,344],'left_turn':[68,264],'right_turn_12':[52,260]}
    for i,(name,motion) in enumerate(MOTIONS.items()):
        accepted.append(dict(name=name,motion=motion,path=f'outputs/retargeted/human39/{name}.npz',
                             category=i,frames=trims[name],split='train'))
    for result in sorted(results,key=lambda x:x['motion']):
        if result.get('quality',{}).get('accepted'):
            for j,frames in enumerate(result['clips']):
                accepted.append(dict(name=result['name']+f'_clip{j:03d}',motion=result['motion'],
                                     path=result['path'],category=result['category'],frames=frames,split='train'))
    manifest={'version':'human39_large_v1','frequency':50,'complete':complete,'accepted':accepted,
              'thresholds':LIMITS,'processed_sources':len(results),'candidate_sources':total,
              'accepted_sources':3+sum(bool(r.get('quality',{}).get('accepted')) for r in results),
              'excluded_before_retarget':excluded,'updated_unix':time.time(),
              'limitations':['train-only, not held-out generalization','geom-center slip proxy, not zero slip',
                             'limited collision bodies; no full self-collision certification',
                             'automatic admission; per-source manual visual review not performed']}
    write_json(ROOT/'configs/large_motion_manifest_v1.json',manifest)
    write_json(REPORT/'status.json',{k:v for k,v in manifest.items() if k!='accepted'})
    print(f"prepared={len(results)}/{total} accepted_sources={manifest['accepted_sources']} clips={len(accepted)} complete={complete}",flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);p.add_argument('--limit',type=int)
    p.add_argument('--retry-errors-only',action='store_true')
    args=p.parse_args();OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(REPORT/'prepare.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=json.loads((ROOT/'configs/paths.json').read_text());splits=json.loads(Path(cfg['manifest']).read_text())
    train=splits['KIT_KINESIS_TRAINING_MOTIONS'];test=set(splits['KIT_KINESIS_TESTING_MOTIONS'])
    assert not set(train)&test
    candidates=[];excluded=[]
    for motion in sorted(train):
        if motion in MOTIONS.values():continue
        if any(word in motion.lower() for word in ('run','support','handrail','beam','egyptian','nordic','steps')):
            excluded.append({'motion':motion,'reason':'outside flat-ground unsupported walking/turning scope'})
        elif motion=='KIT/7/RightTurn07_poses':
            excluded.append({'motion':motion,'reason':'previously rejected slip proxy'})
        else:candidates.append(motion)
    # Interleave subjects to get diverse initial data if interrupted.
    candidates.sort(key=lambda s:(s.split('/')[-1],s.split('/')[1]))
    if args.limit:candidates=candidates[:args.limit]
    write_json(REPORT/'plan.json',dict(candidates=candidates,excluded=excluded,workers=args.workers,
                                     split_manifest_sha256=hashlib.sha256(Path(cfg['manifest']).read_bytes()).hexdigest()))
    results=[];publish(results,excluded,len(candidates))
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(worker,candidates,itertools.repeat(args.retry_errors_only)):
            results.append(result);publish(results,excluded,len(candidates))
    publish(results,excluded,len(candidates),complete=True)

if __name__=='__main__':main()
