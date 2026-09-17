#!/usr/bin/env python3
"""Rebuild the frozen 257-source dataset without distributing motion data."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,concurrent.futures,contextlib,hashlib,json
from pathlib import Path
import numpy as np
from retarget_amass import ROOT,MODEL,retarget

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def inspect(path,expected):
 mismatches=[]
 with np.load(path,allow_pickle=False) as z:
  meta=json.loads(str(z['metadata']))
  assert meta['source_sha256']==expected['source_sha256']
  assert meta['model_sha256']==expected['model_sha256']
  for key,ref in expected['arrays'].items():
   a=z[key]
   assert list(a.shape)==ref['shape'],(path,key,'shape')
   if np.issubdtype(a.dtype,np.number):assert np.isfinite(a).all(),(path,key,'nonfinite')
   if str(a.dtype)!=ref['dtype'] or hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()!=ref['sha256']:mismatches.append(key)
 return mismatches

def worker(item):
 rel,expected=item;path=ROOT/rel
 cfg=json.loads((ROOT/'configs/paths.json').read_text());raw=Path(cfg['raw_kit']).parent/(expected['motion']+'.npz')
 assert digest(raw)==expected['source_sha256'],f'Raw source mismatch: {raw}'
 assert digest(MODEL)==expected['model_sha256'],'Model mismatch'
 if path.exists():
  drift=inspect(path,expected)
 else:
  log=ROOT/'reports/regeneration_logs'/(path.stem+'.log');log.parent.mkdir(parents=True,exist_ok=True)
  with log.open('w') as f,contextlib.redirect_stdout(f):retarget(path.stem,expected['motion'],50,path.parent)
  drift=inspect(path,expected)
 return {'path':rel,'numeric_mismatches':drift,'file_sha256':digest(path)}

def main():
 p=argparse.ArgumentParser();p.add_argument('--kit',type=Path,required=True);p.add_argument('--body-models',type=Path,required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--limit',type=int);p.add_argument('--allow-numeric-drift',action='store_true',help='Explicitly allow different numeric arrays; do not call this an exact data reproduction')
 args=p.parse_args();assert args.kit.name=='KIT','--kit must point to the original KIT directory'
 refs=json.loads((ROOT/'configs/reproduction_assets.json').read_text());manifest=json.loads((ROOT/'configs/large_motion_train_v1.json').read_text());splits=json.loads((ROOT/'configs/source_split.json').read_text())
 train=set(splits['KIT_KINESIS_TRAINING_MOTIONS']);test=set(splits['KIT_KINESIS_TESTING_MOTIONS']);assert not train&test
 for a in refs['assets'].values():assert a['motion'] in train and a['motion'] not in test
 for gender,h in refs['body_model_sha256'].items():assert digest(args.body_models/'smplh'/gender/'model.npz')==h,f'SMPL-H {gender} differs'
 (ROOT/'configs/paths.json').write_text(json.dumps({'raw_kit':str(args.kit.resolve()),'body_models':str(args.body_models.resolve()),'manifest':str(ROOT/'configs/source_split.json')},indent=2))
 items=list(refs['assets'].items());items=items[:args.limit] if args.limit else items
 rows=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
  for row in pool.map(worker,items):
   rows.append(row);print(f"{len(rows)}/{len(items)} {row['path']} drift={row['numeric_mismatches']}",flush=True)
 drift=[r for r in rows if r['numeric_mismatches']]
 (ROOT/'reports').mkdir(exist_ok=True);(ROOT/'reports/regeneration_audit.json').write_text(json.dumps({'files':rows,'numeric_exact':not drift},indent=2))
 if drift and not args.allow_numeric_drift:raise SystemExit('Numeric fingerprints differ: see reports/regeneration_audit.json. No usable manifest emitted. Use the recorded environment; --allow-numeric-drift is an explicit non-exact alternative.')
 hashes={r['path']:r['file_sha256'] for r in rows};entries=[e for e in manifest['accepted'] if e['path'] in hashes]
 for e in entries:e['file_sha256']=hashes[e['path']]
 manifest.update(accepted=entries,complete=len(rows)==len(refs['assets']),regenerated=True,numeric_exact=not drift)
 (ROOT/'configs/regenerated_manifest.json').write_text(json.dumps(manifest,indent=2))
 print(f'Ready: {len(entries)} clips; numeric_exact={not drift}; complete={manifest["complete"]}')
if __name__=='__main__':main()
