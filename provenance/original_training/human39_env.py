"""Reference-conditioned 39-DOF PD-torque imitation, built on IsaacLab DirectRLEnv.
No root forces, kinematic tracking, muscle actuators, or pretrained robot policy.
"""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul, quat_conjugate

ROOT=Path(__file__).resolve().parents[1]
MOTIONS=['walk','left_turn','right_turn_12']

@configclass
class Human39Cfg(DirectRLEnvCfg):
    motion_manifest: str = ''
    decimation=10
    episode_length_s=8.0
    action_space=39
    observation_space=219
    state_space=0
    seed=42
    sim=sim_utils.SimulationCfg(dt=.002,render_interval=10,physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.9,dynamic_friction=.8,restitution=0))
    scene=InteractiveSceneCfg(num_envs=65536,env_spacing=6.0,replicate_physics=True,clone_in_fabric=True)
    robot_cfg=ArticulationCfg(
        prim_path='/World/envs/env_.*/Human',
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(ROOT/'models/processed/human39/human_training.urdf'),
            usd_dir=str(ROOT/'models/processed/human39/usd'),usd_file_name='human39.usd',
            fix_base=False,merge_fixed_joints=True,root_link_name='PelvisLink',
            self_collision=False,collision_from_visuals=False,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(target_type='none',gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0,damping=0)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False,max_depenetration_velocity=1.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False,solver_position_iteration_count=8,solver_velocity_iteration_count=2),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0,0,.96),joint_pos={'.*':0.0}),
        actuators={'torque':ImplicitActuatorCfg(joint_names_expr=['.*'],stiffness=0.0,damping=0.0,effort_limit_sim=300.0,velocity_limit_sim=20.0,armature=.01)},
        soft_joint_pos_limit_factor=1.0,
    )

class Human39Env(DirectRLEnv):
    def __init__(self,cfg,**kwargs):
        super().__init__(cfg,**kwargs)
        self.joint_names=self.robot.joint_names
        assert len(self.joint_names)==39,self.joint_names
        act=json.loads((ROOT/'models/source/human_39dof/human_description/config/actuation_defaults.json').read_text())['joints']
        self.kp=torch.tensor([act[n]['kp'] for n in self.joint_names],dtype=torch.float32,device=self.device)
        self.kd=torch.tensor([act[n]['kd'] for n in self.joint_names],dtype=torch.float32,device=self.device)
        self.effort=torch.tensor([act[n]['effort'] for n in self.joint_names],dtype=torch.float32,device=self.device)
        # More responsive lower-body tracking for this simulation, within original effort bounds.
        self.kp*=2.0;self.kd*=1.4
        self.track_names=['PelvisLink','ChestLink','HeadLink','LeftFootLink','RightFootLink','LeftHandLink','RightHandLink']
        self.track_ids=[self.robot.body_names.index(n) for n in self.track_names]
        motions=[];self.trim_info={}
        if cfg.motion_manifest:
            manifest=json.loads((ROOT/cfg.motion_manifest).read_text())
            entries=manifest['accepted']
            assert entries and all(e['split']=='train' for e in entries)
        else:
            entries=[{'name':n,'path':f'outputs/retargeted/human39/{n}.npz','category':i} for i,n in enumerate(MOTIONS)]
        self.motion_entries=entries
        self.num_motions=len(entries)
        self.motion_categories=torch.tensor([e['category'] for e in entries],device=self.device)
        assert ((self.motion_categories>=0)&(self.motion_categories<3)).all()
        verified_files=set()
        for entry in entries:
            name=entry['name']
            path=ROOT/entry['path']
            if entry.get('file_sha256') and entry['path'] not in verified_files:
                assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['file_sha256'],str(path)
                verified_files.add(entry['path'])
            z=np.load(path)
            assert float(z['frequency'])==50
            order=[list(z['joint_names']).index(n) for n in self.joint_names]
            body_order=[list(z['body_names']).index(n) for n in self.track_names]
            if 'frames' in entry:
                lo,hi=entry['frames']
            else:
                speed=np.linalg.norm(z['qvel'][:,:2],axis=1);moving=np.flatnonzero(speed>.12)
                lo=max(0,int(moving[0])-25);hi=min(len(speed),int(moving[-1])+26)
            assert 0<=lo<hi<=len(z['qpos']) and hi-lo>=50,(name,lo,hi)
            self.trim_info[name]=[lo,hi]
            q=z['qpos'][lo:hi].copy();v=z['qvel'][lo:hi].copy();xyz=z['xpos'][lo:hi,body_order].copy()
            from scipy.spatial.transform import Rotation
            omega=Rotation.from_quat(q[:,3:7],scalar_first=True).apply(v[:,3:6])
            motions.append({'q':q[:,7:][:,order],'v':v[:,6:][:,order],'root':q[:,:7],'rootvel':np.concatenate([v[:,:3],omega],axis=1),'body':xyz})
        self.lengths=torch.tensor([len(m['q']) for m in motions],device=self.device)
        self.offsets=torch.cat([torch.zeros(1,device=self.device,dtype=torch.long),self.lengths.cumsum(0)[:-1]])
        self.refs={}
        for k in motions[0]:
            self.refs[k]=torch.tensor(np.concatenate([m[k] for m in motions]),dtype=torch.float32,device=self.device)
        self.motion_ids=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
        self.start_frames=torch.zeros_like(self.motion_ids)
        self.actions=torch.zeros((self.num_envs,39),device=self.device);self.previous_actions=self.actions.clone();self.torques=self.actions.clone()
        self.target_q=self.actions.clone();self.target_v=self.actions.clone()
        self._audit_written=False
        print('HUMAN39_JOINT_ORDER',self.joint_names,flush=True)
        print('HUMAN39_TRIMS',self.trim_info,flush=True)

    def _setup_scene(self):
        self.robot=Articulation(self.cfg.robot_cfg)
        # Cover the entire clone grid plus full motion paths, at every scale.
        ground_width=2*(int(np.ceil(np.sqrt(self.cfg.scene.num_envs)))+2)*self.cfg.scene.env_spacing
        ground=sim_utils.CuboidCfg(size=(ground_width,ground_width,.1),collision_props=sim_utils.CollisionPropertiesCfg(),physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.9,dynamic_friction=.8,restitution=0),visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.65,.68,.7)))
        ground.func('/World/ground',ground,translation=(0,0,-.05))
        self.scene.clone_environments(copy_from_source=False)
        # GPU replication already uses environment IDs to filter collisions.
        # Authoring per-environment USD collision groups again is redundant and
        # scales poorly. The global static ground remains visible to every ID.
        if self.device=='cpu' or not self.cfg.scene.replicate_physics:
            self.scene.filter_collisions(global_prim_paths=['/World/ground'])
        self.scene.articulations['human']=self.robot
        light=sim_utils.DomeLightCfg(intensity=1800,color=(.8,.8,.8));light.func('/World/light',light)

    def ref(self,key,next_frame=False):
        index=torch.minimum(self.start_frames+self.episode_length_buf+int(next_frame),self.lengths[self.motion_ids]-1)
        return self.refs[key][self.offsets[self.motion_ids]+index]

    def _pre_physics_step(self,actions):
        self.previous_actions=self.actions.clone();self.actions=actions.clamp(-1,1)
        self.target_q=self.ref('q',True)+.3*self.actions
        limits=self.robot.data.soft_joint_pos_limits
        self.target_q=self.target_q.clamp(limits[:,:,0],limits[:,:,1])
        self.target_v=self.ref('v',True)

    def _apply_action(self):
        self.torques=(self.kp*(self.target_q-self.robot.data.joint_pos)+self.kd*(self.target_v-self.robot.data.joint_vel)).clamp(-self.effort,self.effort)
        self.robot.set_joint_effort_target(self.torques)

    def _get_observations(self):
        data=self.robot.data;root=self.ref('root');vel=self.ref('rootvel');q=data.root_quat_w
        err=quat_apply_inverse(q,root[:,:3]+self.scene.env_origins-data.root_pos_w)
        rot_err=quat_mul(quat_conjugate(q),root[:,3:7]);rot_err*=torch.where(rot_err[:,:1]<0,-1.,1.)
        phase=(self.start_frames+self.episode_length_buf)/self.lengths[self.motion_ids]
        obs=torch.cat([data.root_lin_vel_b,data.root_ang_vel_b,data.projected_gravity_b,data.joint_pos,data.joint_vel*.1,self.actions,self.ref('q'),self.ref('v')*.1,err,rot_err,quat_apply_inverse(q,vel[:,:3]),torch.stack([torch.sin(phase*2*torch.pi),torch.cos(phase*2*torch.pi)],-1),torch.nn.functional.one_hot(self.motion_categories[self.motion_ids],3).float()],dim=-1)
        return {'policy':obs}

    def _errors(self):
        d=self.robot.data;root=self.ref('root')
        qe=((d.joint_pos-self.ref('q'))**2).mean(-1)
        ve=((d.joint_vel-self.ref('v'))**2).mean(-1)
        pos=d.body_pos_w[:,self.track_ids]-self.scene.env_origins[:,None,:]
        be=((pos-self.ref('body'))**2).sum(-1).mean(-1)
        re=((d.root_pos_w-self.scene.env_origins-root[:,:3])**2).sum(-1)
        oe=1-(d.root_quat_w*root[:,3:7]).sum(-1).square().clamp(0,1)
        return qe,ve,be,re,oe

    def _get_rewards(self):
        qe,ve,be,re,oe=self._errors()
        reward=.35*torch.exp(-qe/.09)+.10*torch.exp(-ve/9)+.25*torch.exp(-be/.04)+.15*torch.exp(-re/.04)+.15*torch.exp(-oe/.08)
        reward-=.01*((self.actions-self.previous_actions)**2).mean(-1)+.003*((self.torques/self.effort)**2).mean(-1)
        reward-=.3*self.reset_terminated.float()
        self.extras['log']={'tracking/joint_rmse':qe.mean().sqrt(),'tracking/body_rmse':be.mean().sqrt(),'tracking/root_rmse':re.mean().sqrt(),'control/torque_saturation':(self.torques.abs()>=self.effort*.99).float().mean(),'tracking/reward':reward.mean()}
        return reward

    def _get_dones(self):
        d=self.robot.data;root=self.ref('root');delta=d.root_pos_w-self.scene.env_origins-root[:,:3]
        failed=(d.root_pos_w[:,2]<.50)|(d.projected_gravity_b[:,2]>-.35)|(delta.norm(dim=-1)>.85)|(~torch.isfinite(d.joint_pos).all(-1))
        timeout=(self.start_frames+self.episode_length_buf>=self.lengths[self.motion_ids]-2)|(self.episode_length_buf>=self.max_episode_length-1)
        return failed,timeout

    def _reset_idx(self,env_ids):
        if env_ids is None:env_ids=self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        self.motion_ids[env_ids]=torch.randint(0,self.num_motions,(len(env_ids),),device=self.device)
        # Reference state initialization; leave at least one second of motion.
        self.start_frames[env_ids]=(torch.rand(len(env_ids),device=self.device)*(self.lengths[self.motion_ids[env_ids]]-50).clamp(min=1)).long()
        root=self.ref('root')[env_ids].clone();root[:,:3]+=self.scene.env_origins[env_ids];root[:,2]+=.003
        self.robot.write_root_pose_to_sim(root,env_ids)
        self.robot.write_root_velocity_to_sim(self.ref('rootvel')[env_ids],env_ids)
        self.robot.write_joint_state_to_sim(self.ref('q')[env_ids],self.ref('v')[env_ids],None,env_ids)
        self.actions[env_ids]=0;self.previous_actions[env_ids]=0
