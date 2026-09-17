#!/usr/bin/env python3
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse
from pathlib import Path
import numpy as np,mujoco,imageio.v2 as imageio
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('evaluation',type=Path);p.add_argument('--envs',type=int,nargs='+');args=p.parse_args();z=np.load(args.evaluation)
m=mujoco.MjModel.from_xml_path(str(ROOT/'models/source/human_39dof/mujoco/human.xml'));d=mujoco.MjData(m)
order=[list(z['joint_names']).index(m.joint(i).name) for i in range(1,m.njnt)]
r=mujoco.Renderer(m,480,640);c=mujoco.MjvCamera();c.azimuth=135;c.elevation=-12;c.distance=3.2
for e in (args.envs if args.envs is not None else range(min(z['qpos'].shape[1],3))):
 frames=z['qpos'][:,e][z['active'][:,e]];snaps=[];indices=np.unique(np.linspace(0,len(frames)-1,6,dtype=int));out=args.evaluation.parent
 with imageio.get_writer(out/f'policy_motion_{e}.mp4',fps=25,codec='libx264',quality=8) as writer:
  for i,q in enumerate(frames):
   d.qpos[:7]=q[:7];d.qpos[7:]=q[7:][order];mujoco.mj_forward(m,d);c.lookat[:]=q[:3];c.lookat[2]=.8
   if i%2==0 or i in indices:
    r.update_scene(d,c);im=r.render()
    if i%2==0:writer.append_data(im)
    if i in indices:snaps.append(im.copy())
 if len(snaps)==6:imageio.imwrite(out/f'policy_motion_{e}.jpg',np.concatenate([np.concatenate(snaps[:3],axis=1),np.concatenate(snaps[3:],axis=1)],axis=0))
r.close()
