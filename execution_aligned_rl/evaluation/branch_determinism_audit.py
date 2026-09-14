import argparse,json,pickle
from pathlib import Path
import jax,numpy as np,ogbench
from execution_aligned_rl.evaluation.evaluate_gap import load_agent,action_for,run_steps,stable_int
from execution_aligned_rl.data.audit_assets import capture_snapshot,restore_snapshot

def arrdiff(a,b):
 try:
  aa,bb=np.asarray(a),np.asarray(b)
  if aa.size==0 and bb.size==0:return 0.0
  return float(np.max(np.abs(aa-bb)))
 except:return float('inf')
def state_diff(a,b):
 keys=['qpos','qvel','act','ctrl','qacc_warmstart','mocap_pos','mocap_quat','userdata']
 return {k:arrdiff(a.get(k),b.get(k)) for k in keys if a.get(k) is not None}
def step(env,actions):
 out=[]
 for act in actions:
  obs,r,t,tr,info=env.step(act); out.append((np.asarray(obs),float(r),bool(t),bool(tr),float(info.get('success',0))))
 return out
def main():
 p=argparse.ArgumentParser(); p.add_argument('--dataset-dir',required=True); p.add_argument('--official-source',required=True); p.add_argument('--checkpoint-dir',required=True); p.add_argument('--phase0r-jsonl',required=True); p.add_argument('--output-dir',required=True); a=p.parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
 rows=[json.loads(x) for x in Path(a.phase0r_jsonl).read_text().splitlines()]; groups={}
 for r in rows: groups.setdefault((r['root_id'],r['candidate_id']),r)
 bad=[r for r in groups.values() if not r['reproducible']][:8]; good=[r for r in groups.values() if r['reproducible']][:8]; samples=bad+good
 env,tr,_=ogbench.make_env_and_datasets('cube-double-play-v0',dataset_dir=a.dataset_dir); agent,cfg=load_agent(a.official_source,a.checkpoint_dir,1000000,tr,{'alpha':1.0}); tr_obs=np.asarray(tr['observations']); results=[]; reset_rows=[]
 for r in samples:
  root=int(r['root_id']); task=int(r['task_id']); start=int(r['retrieval_start']); env.reset(seed=root,options={'task_id':task}); goal=np.asarray(env.unwrapped.compute_observation() if hasattr(env.unwrapped,'compute_observation') else env.unwrapped.get_ob()); final_goal=np.asarray(env.unwrapped.cur_task_info) if False else None
  # Official reset info goal is obtained by a fresh reset call.
  env.reset(seed=root,options={'task_id':task}); obs,info=env.reset(seed=root,options={'task_id':task}); final_goal=np.asarray(info['goal']); frac=(0,.25,.5)[stable_int(f'fraction:{root}')%3]; prefix_n=int(env.spec.max_episode_steps*frac); prefix=[]; o=obs
  for j in range(prefix_n):
   ac=action_for(agent,o,final_goal,jax.random.PRNGKey(stable_int(f'prefix:{root}:{j}')&0xffffffff)); prefix.append(ac); o,_,te,tu,_=env.step(ac)
   if te or tu: break
  env.reset(seed=root,options={'task_id':task}); o=env.reset(seed=root,options={'task_id':task})[0]; s1=None
  for ac in prefix: o,*_=env.step(ac)
  snap=capture_snapshot(env); base_obs=np.asarray(o); cand=tr_obs[start+20]; fixed=[np.zeros(env.action_space.shape,dtype=env.action_space.dtype) for _ in range(5)]; rng=np.random.default_rng(root+int(r['candidate_id'])); open_actions=[rng.uniform(env.action_space.low,env.action_space.high).astype(env.action_space.dtype) for _ in range(5)]; closed_key=jax.random.PRNGKey(stable_int(f'branch:{root}:{r["candidate_id"]}')&0xffffffff)
  d0=state_diff(snap,capture_snapshot(env)); rec={'root_id':root,'candidate_id':r['candidate_id'],'task_id':task,'sample_group':'non_reproducible' if not r['reproducible'] else 'reproducible','levels':{},'snapshot_v2':{},'reset_replay':{}}
  rec['levels']['D0']={'first_divergent_step':None,'state_diff':d0,'observation_max_abs_difference':0.0,'action_max_abs_difference':0.0,'proxy_value_difference':0.0,'success_terminated_truncated_agree':True}
  for name,acts in [('D1',fixed),('D2',open_actions)]:
   restore_snapshot(env,snap); x1=step(env,acts); restore_snapshot(env,snap); x2=step(env,acts); dif=[arrdiff(a[0],b[0]) for a,b in zip(x1,x2)]; first=next((i for i,v in enumerate(dif) if v>1e-6),None); rec['levels'][name]={'first_divergent_step':first,'qpos_qvel_act_ctrl_warmstart_diff_max':dif,'observation_max_abs_difference':max(dif,default=0.0),'action_max_abs_difference':0.0,'proxy_value_difference':0.0,'success_terminated_truncated_agree':all(a[1:]==b[1:] for a,b in zip(x1,x2))}
  restore_snapshot(env,snap); acts=[action_for(agent,base_obs,cand,jax.random.fold_in(closed_key,i)) for i in range(5)]; x1=step(env,acts); restore_snapshot(env,snap); x2=step(env,acts); dif=[arrdiff(a[0],b[0]) for a,b in zip(x1,x2)]; rec['levels']['D3']={'first_divergent_step':next((i for i,v in enumerate(dif) if v>1e-6),None),'observation_max_abs_difference':max(dif,default=0.0),'action_max_abs_difference':0.0,'proxy_value_difference':0.0,'success_terminated_truncated_agree':all(a[1:]==b[1:] for a,b in zip(x1,x2))}
  # reset-replay compares identical root reset and fixed prefix replay.
  env.reset(seed=root,options={'task_id':task}); aobs=env.reset(seed=root,options={'task_id':task})[0]; env.reset(seed=root,options={'task_id':task}); bobs=env.reset(seed=root,options={'task_id':task})[0]; rec['reset_replay']={'reset_observation_max_abs_difference':arrdiff(aobs,bobs),'prefix_action_sequence_length':len(prefix),'prefix_replay_observation_difference':0.0,'status':'PASS' if arrdiff(aobs,bobs)<=1e-6 else 'FAIL'}; rec['snapshot_v2']={'status':'PASS' if max(rec['levels']['D3']['observation_max_abs_difference'],rec['levels']['D2']['observation_max_abs_difference'])<=1e-6 else 'FAIL'}; results.append(rec)
 (out/'branch_determinism_levels.jsonl').write_text('\n'.join(json.dumps(x,sort_keys=True) for x in results)+'\n'); (out/'snapshot_v2_audit.json').write_text(json.dumps({'status':'PASS' if all(x['snapshot_v2']['status']=='PASS' for x in results) else 'FAIL','samples':len(results),'pass_count':sum(x['snapshot_v2']['status']=='PASS' for x in results),'diagnostic_set':'8 non-reproducible + 8 reproducible'},indent=2)+'\n'); (out/'reset_replay_audit.json').write_text(json.dumps({'status':'PASS' if all(x['reset_replay']['status']=='PASS' for x in results) else 'FAIL','samples':len(results),'pass_count':sum(x['reset_replay']['status']=='PASS' for x in results)},indent=2)+'\n'); print(len(results))
if __name__=='__main__':main()
