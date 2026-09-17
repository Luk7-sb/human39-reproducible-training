#!/usr/bin/env python3
"""SMPL-H joint FK -> bounded whole-body IK for the supplied human39 model.
Only joint centers are needed: DMPL skin deformation does not change this skeleton.
All source rotations come from AMASS; target joint coordinates are solved afresh.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import json, argparse, hashlib
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT/'models/source/human_39dof/mujoco/human.xml'
C = np.array([[0,0,1],[1,0,0],[0,1,0.]])
MOTIONS = {'walk':'KIT/314/walking_medium09_poses', 'left_turn':'KIT/12/LeftTurn03_poses', 'right_turn_12':'KIT/12/RightTurn03_poses'}

def decode(path, body_root, fps):
    z=np.load(path,allow_pickle=False)
    gender=z['gender'].item()
    if isinstance(gender,bytes): gender=gender.decode()
    bm=np.load(body_root/'smplh'/gender/'model.npz',allow_pickle=False)
    betas=z['betas'].reshape(-1)[:16]
    verts=bm['v_template']+np.einsum('vck,k->vc',bm['shapedirs'][...,:len(betas)],betas)
    rest=(bm['J_regressor']@verts)[:22]
    parent=bm['kintree_table'][0,:22].astype(int);parent[0]=-1
    src_fps=float(z['mocap_framerate']); t=np.arange(len(z['poses']))/src_fps
    # Floating-point arange can overshoot the source endpoint by one ulp.
    times=np.minimum(np.arange(0,t[-1]+1e-9,1/fps),t[-1])
    local=np.stack([Slerp(t,R.from_rotvec(z['poses'][:,3*j:3*j+3]))(times).as_matrix() for j in range(22)],axis=1)
    trans=np.stack([np.interp(times,t,z['trans'][:,i]) for i in range(3)],axis=1)
    global_r=np.zeros_like(local);joints=np.zeros((len(times),22,3))
    for j in range(22):
        if j==0: global_r[:,j]=local[:,j];joints[:,j]=rest[j]+trans
        else:
            global_r[:,j]=global_r[:,parent[j]]@local[:,j]
            joints[:,j]=joints[:,parent[j]]+np.einsum('tij,j->ti',global_r[:,parent[j]],rest[j]-rest[parent[j]])
    return joints,global_r,{'gender':gender,'source_fps':src_fps,'source_frames':len(t),'frames':len(times),'dmpl':'not used: joint skeleton only'}

def unit(v): return v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-9)

def retarget(key, motion, fps=50, output_dir=None):
    cfg=json.loads((ROOT/'configs/paths.json').read_text()); src=Path(cfg['raw_kit']).parent/(motion+'.npz')
    source,rots,meta=decode(src,Path(cfg['body_models']),fps)
    m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    p0={m.body(i).name:d.xpos[i].copy() for i in range(1,m.nbody)}
    names=['PelvisLink','TorsoLink','ChestLink','HeadLink','LeftThighLink','LeftShinLink','LeftFootLink','RightThighLink','RightShinLink','RightFootLink','LeftUpperArmLink','LeftForearmLink','LeftHandLink','RightUpperArmLink','RightForearmLink','RightHandLink']
    ids=[m.body(n).id for n in names]
    rest_leg=np.linalg.norm(p0['LeftShinLink']-p0['LeftThighLink'])+np.linalg.norm(p0['LeftFootLink']-p0['LeftShinLink'])
    src_leg=np.linalg.norm(source[0,4]-source[0,1])+np.linalg.norm(source[0,7]-source[0,4]);scale=rest_leg/src_leg
    rootrot=rots[:,0]@C.T
    rootpos=(source[:,0]-source[0,0])*scale+p0['PelvisLink']
    targets=[];orientations=[]
    for f in range(len(source)):
        rr=rootrot[f];s=source[f];gr=rots[f]
        t={'PelvisLink':rootpos[f]}
        t['TorsoLink']=t['PelvisLink']+rr@(p0['TorsoLink']-p0['PelvisLink'])
        torso=gr[3]@C.T;chest=gr[9]@C.T
        t['ChestLink']=t['TorsoLink']+torso@(p0['ChestLink']-p0['TorsoLink'])
        t['HeadLink']=t['ChestLink']+chest@(p0['HeadLink']-p0['ChestLink'])
        for side,hip,knee,ankle,shoulder,elbow,wrist in [('Left',1,4,7,16,18,20),('Right',2,5,8,17,19,21)]:
            thigh,shin,foot=[side+x for x in ['ThighLink','ShinLink','FootLink']]
            upper,fore,hand=[side+x for x in ['UpperArmLink','ForearmLink','HandLink']]
            t[thigh]=t['PelvisLink']+rr@(p0[thigh]-p0['PelvisLink'])
            t[shin]=t[thigh]+unit(s[knee]-s[hip])*np.linalg.norm(p0[shin]-p0[thigh])
            t[foot]=t[shin]+unit(s[ankle]-s[knee])*np.linalg.norm(p0[foot]-p0[shin])
            t[upper]=t['ChestLink']+chest@(p0[upper]-p0['ChestLink'])
            t[fore]=t[upper]+unit(s[elbow]-s[shoulder])*np.linalg.norm(p0[fore]-p0[upper])
            t[hand]=t[fore]+unit(s[wrist]-s[elbow])*np.linalg.norm(p0[hand]-p0[fore])
        targets.append(np.array([t[n] for n in names]))
        orientations.append([rr,torso,chest,gr[15]@C.T,gr[7]@C.T,gr[8]@C.T])
    targets=np.array(targets);orientations=np.array(orientations)
    ori_ids=[m.body(n).id for n in ['PelvisLink','TorsoLink','ChestLink','HeadLink','LeftFootLink','RightFootLink']]
    ori_weight=[.8,.15,.35,.12,.25,.25]
    pos_weight=np.array([3,1,1,1,1,3,5,1,3,5,1,2,3,1,2,3])
    qp=[];errors=[];jp=np.zeros((3,m.nv));jr=np.zeros_like(jp)
    ranges=m.jnt_range[1:];prev=d.qpos.copy()
    for f in range(len(source)):
        d.qpos[:3]=rootpos[f];d.qpos[3:7]=R.from_matrix(rootrot[f]).as_quat(scalar_first=True)
        if f: d.qpos[7:]=prev[7:]
        for iteration in range(40 if f==0 else 12):
            mujoco.mj_forward(m,d);rows=[];rhs=[]
            for n,b in enumerate(ids):
                mujoco.mj_jacBody(m,d,jp,jr,b);rows.append(jp.copy()*pos_weight[n]);rhs.append((targets[f,n]-d.xpos[b])*pos_weight[n])
            for n,b in enumerate(ori_ids):
                mujoco.mj_jacBody(m,d,jp,jr,b);rows.append(jr.copy()*ori_weight[n]);rhs.append(R.from_matrix(orientations[f,n]@d.xmat[b].reshape(3,3).T).as_rotvec()*ori_weight[n])
            # Mild posture and temporal regularization resolve unobserved axes.
            reg=np.zeros((39,m.nv));reg[:,6:]=np.eye(39)*.025
            rows.append(reg);rhs.append(-d.qpos[7:]*.025)
            if f: rows.append(reg*2);rhs.append((prev[7:]-d.qpos[7:])*.05)
            A=np.concatenate(rows);e=np.concatenate(rhs)
            dq=np.linalg.solve(A.T@A+np.eye(m.nv)*.002,A.T@e)
            dq=np.clip(dq,-.15,.15);mujoco.mj_integratePos(m,d.qpos,dq,1)
            d.qpos[7:]=np.clip(d.qpos[7:],ranges[:,0]+1e-5,ranges[:,1]-1e-5)
            if np.linalg.norm(dq)<1e-5:break
        mujoco.mj_forward(m,d);errors.append(np.linalg.norm(d.xpos[ids]-targets[f],axis=1));qp.append(d.qpos.copy());prev=d.qpos.copy()
        if f%100==0: print(key,f,'/',len(source),flush=True)
    qp=np.array(qp)
    # Smooth joint coordinates, then lift root using actual split-foot collision corners.
    qp[:,7:]=gaussian_filter1d(qp[:,7:],.65,axis=0)
    feet=[i for i in range(m.ngeom) if m.geom_contype[i] and 'Foot' in m.body(m.geom_bodyid[i]).name or m.geom_contype[i] and 'Forefoot' in m.body(m.geom_bodyid[i]).name]
    corners=np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])
    min_z=[]
    for q in qp:
        d.qpos[:]=q;mujoco.mj_forward(m,d)
        min_z.append(min((d.geom_xpos[g]+(corners*m.geom_size[g])@d.geom_xmat[g].reshape(3,3).T)[:,2].min() for g in feet))
    correction=gaussian_filter1d(-np.array(min_z),1)
    qp[:,2]+=correction
    # Normalize initial heading and XY origin without changing motion or turn sign.
    yaw=R.from_matrix(rootrot[0]).as_euler('ZYX')[0]; rz=R.from_euler('z',-yaw)
    qp[:,:3]-=np.array([qp[0,0],qp[0,1],0]);qp[:,:3]=rz.apply(qp[:,:3]);qp[:,3:7]=(rz*R.from_quat(qp[:,3:7],scalar_first=True)).as_quat(scalar_first=True)
    for f in range(1,len(qp)):
        if np.dot(qp[f,3:7],qp[f-1,3:7])<0:qp[f,3:7]*=-1
    qv=np.zeros((len(qp),m.nv))
    for f in range(len(qp)):
        a=max(0,f-1);b=min(len(qp)-1,f+1);mujoco.mj_differentiatePos(m,qv[f],(b-a)/fps,qp[a],qp[b])
    xpos=[];xquat=[]
    for q in qp:
        d.qpos[:]=q;mujoco.mj_forward(m,d);xpos.append(d.xpos.copy());xquat.append(d.xquat.copy())
    meta.update(motion=motion,output_fps=fps,scale=float(scale),model_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),ik_mean_position_error_m=float(np.mean(errors)),ik_max_position_error_m=float(np.max(errors)),ground_correction_range_m=[float(correction.min()),float(correction.max())],method='SMPL-H shaped joint FK; anthropometric chain targets; bounded damped IK; split-foot ground correction')
    out=Path(output_dir) if output_dir is not None else ROOT/'outputs/retargeted/human39';out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/(key+'.npz'),qpos=qp,qvel=qv,xpos=xpos,xquat=xquat,frequency=fps,joint_names=[m.joint(i).name for i in range(1,m.njnt)],body_names=[m.body(i).name for i in range(m.nbody)],metadata=json.dumps(meta))
    (out/(key+'.json')).write_text(json.dumps(meta,indent=2));print(meta,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--motion',choices=list(MOTIONS)+['all'],default='all');p.add_argument('--fps',type=int,default=50);args=p.parse_args()
    for key,motion in MOTIONS.items():
        if args.motion in ('all',key):retarget(key,motion,args.fps)
