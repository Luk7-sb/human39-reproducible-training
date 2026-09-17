#!/usr/bin/env python3
import argparse,json,time
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--num_envs',type=int,default=65536);p.add_argument('--iterations',type=int,default=1000);p.add_argument('--run',default='pd_3motions');p.add_argument('--checkpoint');p.add_argument('--smoke',action='store_true');p.add_argument('--eval_steps',type=int,default=0)
p.add_argument('--minibatches',type=int,default=0,help='0: cap each PPO minibatch near 32768 samples')
p.add_argument('--torch_threads',type=int,default=8)
p.add_argument('--save_interval',type=int,default=50)
p.add_argument('--motion_manifest',default='')
p.add_argument('--max_hours',type=float,default=0,help='Save and end training at an iteration boundary after this wall time')
AppLauncher.add_app_launcher_args(p);args=p.parse_args();app=AppLauncher(args).app
import numpy as np
import torch
torch.set_num_threads(args.torch_threads)
from human39_env import Human39Cfg,Human39Env,ROOT
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
cfg=Human39Cfg();cfg.scene.num_envs=args.num_envs;cfg.sim.device=args.device;cfg.motion_manifest=args.motion_manifest
# PhysX GPU contact buffers cannot grow automatically. Budget conservatively
# for large clone grids without changing solver fidelity or control frequency.
cfg.sim.physx.gpu_max_rigid_patch_count=max(5*2**15,2**int(np.ceil(np.log2(args.num_envs*32))))
if args.num_envs>=16384:
    cfg.sim.physx.gpu_heap_capacity=2**28
    cfg.sim.physx.gpu_temp_buffer_capacity=2**26
run=ROOT/'runs'/args.run;run.mkdir(parents=True,exist_ok=True)
(run/'arguments.json').write_text(json.dumps(vars(args),indent=2,default=str))
env=Human39Env(cfg);wrapped=RslRlVecEnvWrapper(env,clip_actions=1.0)
(run/'motion_entries.json').write_text(json.dumps(env.motion_entries,indent=2))
if args.motion_manifest:
    (run/'motion_manifest.json').write_text((ROOT/args.motion_manifest).read_text())
train_cfg={'num_steps_per_env':32,'save_interval':100,'obs_groups':{'policy':['policy'],'critic':['policy']},'logger':'tensorboard','policy':{'class_name':'ActorCritic','init_noise_std':.3,'actor_hidden_dims':[256,256,128],'critic_hidden_dims':[256,256,128],'activation':'elu','actor_obs_normalization':True,'critic_obs_normalization':True},'algorithm':{'class_name':'PPO','value_loss_coef':1.,'use_clipped_value_loss':True,'clip_param':.2,'entropy_coef':.002,'num_learning_epochs':5,'num_mini_batches':4,'learning_rate':3e-4,'schedule':'adaptive','gamma':.99,'lam':.95,'desired_kl':.01,'max_grad_norm':1.}}
train_cfg['algorithm']['num_mini_batches']=args.minibatches or max(4,args.num_envs//1024)
train_cfg['save_interval']=args.save_interval
(run/'train_config.json').write_text(json.dumps(train_cfg,indent=2));(run/'joint_mapping.json').write_text(json.dumps({'joint_names':env.joint_names,'trim_frames':env.trim_info,'kp':env.kp.tolist(),'kd':env.kd.tolist(),'effort_limits':env.effort.tolist()},indent=2))
# Validate root/joint mapping with a forward physics update before learning.
initial_body_error=env._errors()[2].mean().sqrt().item()
(run/'import_validation.json').write_text(json.dumps({'initial_body_rmse_m':initial_body_error,'total_mass_kg':float(env.robot.root_physx_view.get_masses()[0].sum())},indent=2))
assert initial_body_error<.015, f'Cross-simulator FK mismatch: {initial_body_error}'
print('HUMAN39_READY',env.num_envs,env.robot.num_joints,wrapped.get_observations()['policy'].shape,flush=True)
if args.smoke:
    rewards=[];dones=0
    with torch.inference_mode():
        for i in range(150):
            obs,r,d,info=wrapped.step(torch.zeros((env.num_envs,39),device=env.device));assert torch.isfinite(obs['policy']).all();assert torch.isfinite(r).all();rewards.append(float(r.mean()));dones+=int(d.sum())
    report={'steps':150,'num_envs':env.num_envs,'finite':True,'mean_reward':float(np.mean(rewards)),'resets':dones,'gpu':torch.cuda.get_device_name(),'observation_dim':obs['policy'].shape[-1]}
    (run/'smoke.json').write_text(json.dumps(report,indent=2));print('HUMAN39_SMOKE',report,flush=True)
else:
    class TrainingBudgetReached(Exception):pass
    training_started=time.monotonic()
    class MeasuredRunner(OnPolicyRunner):
        def log(self,locs,*a,**kw):
            super().log(locs,*a,**kw)
            elapsed=locs['collection_time']+locs['learn_time']
            row={'time':time.time(),'iteration':locs['it'],'num_envs':env.num_envs,'collection_s':locs['collection_time'],'learning_s':locs['learn_time'],'env_steps_s':env.num_envs*self.num_steps_per_env/elapsed,'torch_peak_memory_gib':torch.cuda.max_memory_allocated()/2**30}
            row['last_step_metrics']={k:float(v) for k,v in env.extras.get('log',{}).items()}
            assert all(np.isfinite(v) for v in row['last_step_metrics'].values()),'Nonfinite training metrics'
            with (run/'performance.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if not args.eval_steps and args.max_hours and time.monotonic()-training_started>=args.max_hours*3600:
                self.current_learning_iteration=locs['it']
                self.save(str(run/f"model_{locs['it']}.pt"))
                raise TrainingBudgetReached()
    runner=MeasuredRunner(wrapped,train_cfg,log_dir=str(run),device=env.device)
    if args.checkpoint:runner.load(args.checkpoint)
    if args.eval_steps:
        policy=runner.get_inference_policy(device=env.device)
        # Full-clip evaluation from each motion's trimmed beginning. Report only
        # the first episode so automatic reset cannot masquerade as survival.
        ids=torch.arange(env.num_envs,device=env.device)
        env.motion_ids[:]=ids%env.num_motions;env.start_frames[:]=0;env.episode_length_buf[:]=0
        root=env.ref('root').clone();root[:,:3]+=env.scene.env_origins;root[:,2]+=.003
        env.robot.write_root_pose_to_sim(root,ids)
        env.robot.write_root_velocity_to_sim(env.ref('rootvel'),ids)
        env.robot.write_joint_state_to_sim(env.ref('q'),env.ref('v'),None,ids)
        env.scene.write_data_to_sim();env.sim.forward()
        obs=wrapped.get_observations();states=[];rewards=[];done_count=0
        active=torch.ones(env.num_envs,dtype=torch.bool,device=env.device)
        lengths=torch.zeros(env.num_envs,dtype=torch.long,device=env.device)
        completed=torch.zeros_like(active);score=torch.zeros(env.num_envs,device=env.device)
        initial_motion_ids=env.motion_ids.clone();alive=[]
        with torch.inference_mode():
            for i in range(args.eval_steps):
                states.append(torch.cat([env.robot.data.root_pos_w-env.scene.env_origins,env.robot.data.root_quat_w,env.robot.data.joint_pos],-1).cpu().numpy())
                alive.append(active.cpu().numpy().copy())
                obs,r,d,info=wrapped.step(policy(obs));score+=r*active;lengths+=active.long()
                finished=active&d.bool();completed|=finished&~env.reset_terminated
                active&=~d.bool()
                if not active.any():break
        np.savez_compressed(run/'evaluation.npz',qpos=states,active=alive,joint_names=env.joint_names,frequency=50,motion_ids=initial_motion_ids.cpu().numpy())
        report={'steps':i+1,'num_envs':env.num_envs,'full_clip_completion_rate':float(completed.float().mean()),'per_env':[{'motion_id':int(initial_motion_ids[j]),'survived_s':float(lengths[j])*.02,'completed':bool(completed[j]),'mean_reward':float(score[j]/lengths[j].clamp(min=1))} for j in range(env.num_envs)]}
        (run/'evaluation.json').write_text(json.dumps(report,indent=2));print('HUMAN39_EVALUATION',report,flush=True)
    else:
        stop_reason='iterations'
        try:runner.learn(num_learning_iterations=args.iterations,init_at_random_ep_len=False)
        except TrainingBudgetReached:stop_reason='wall_time_budget'
        (run/'finished.json').write_text(json.dumps({'time':time.time(),'requested_iterations':args.iterations,'last_iteration':runner.current_learning_iteration,'stop_reason':stop_reason}))
wrapped.close();app.close()
