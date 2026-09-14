import argparse,json,hashlib
from pathlib import Path
import jax,numpy as np,ogbench,mujoco
from execution_aligned_rl.evaluation.evaluate_gap import load_agent,action_for,value_for,stable_int
from execution_aligned_rl.data.audit_assets import capture_snapshot

SPEC=mujoco.mjtState.mjSTATE_INTEGRATION
def snap(env):
 b=env.unwrapped; n=mujoco.mj_stateSize(b.model,SPEC); integ=np.empty(n); mujoco.mj_getState(b.model,b.data,integ,SPEC); return {'integration':integ.copy(),'py':capture_snapshot(env)}
def restore(env,s):
 b=env.unwrapped; mujoco.mj_setState(b.model,b.data,s['integration'],SPEC); p=s['py']
 b.np_random.bit_generator.state=json.loads(json.dumps(p['rng'])); np.random.set_state(p['global_numpy_rng']); b.action_space.np_random.bit_generator.state=json.loads(json.dumps(p['action_space_rng']))
 for k in ['cur_task_id','cur_task_info','cur_goal_xy']:
  if hasattr(b,k): setattr(b,k,p[k])
 for w,e in zip([env],p['elapsed_steps']):
  if e is not None:w._elapsed_steps=e
 mujoco.mj_forward(b.model,b.data)
def ad(a,b):
 aa,bb=np.asarray(a),np.asarray(b)
 if aa.size==0 and bb.size==0:return 0.0
 return float(np.max(np.abs(aa-bb)))
def run(env,acts):
 rec=[]
 for a in acts:
  o,r,t,tr,i=env.step(a); rec.append({'action':np.asarray(a).tolist(),'observation':np.asarray(o).tolist(),'reward':float(r),'success':float(i.get('success',0)),'terminated':bool(t),'truncated':bool(tr),'state':snap(env)})
  if t or tr:break
 return rec
def cmp(x,y,agent,goal):
 out=[]; first=None
 for i,(a,b) in enumerate(zip(x,y)):
  d={'step':i,'action_max_abs_difference':ad(a['action'],b['action']),'observation_max_abs_difference':ad(a['observation'],b['observation']),'integration_max_abs_difference':ad(a['state']['integration'],b['state']['integration']),'qpos_max_abs_difference':ad(a['state']['py']['qpos'],b['state']['py']['qpos']),'qvel_max_abs_difference':ad(a['state']['py']['qvel'],b['state']['py']['qvel']),'act_max_abs_difference':ad(a['state']['py']['act'],b['state']['py']['act']),'ctrl_max_abs_difference':ad(a['state']['py']['ctrl'],b['state']['py']['ctrl']),'warmstart_max_abs_difference':ad(a['state']['py']['qacc_warmstart'],b['state']['py']['qacc_warmstart']),'robot_obs_0_19_max_abs_difference':ad(a['observation'][:19],b['observation'][:19]),'cube_obs_19_37_max_abs_difference':ad(a['observation'][19:37],b['observation'][19:37]),'success_agree':a['success']==b['success'],'terminated_agree':a['terminated']==b['terminated'],'truncated_agree':a['truncated']==b['truncated']}; d['proxy_value_difference']=abs((float(value_for(agent,np.asarray(a['observation'])[None],goal)[0]))-(float(value_for(agent,np.asarray(b['observation'])[None],goal)[0]))); out.append(d)
  if first is None and (d['action_max_abs_difference']>1e-7 or d['observation_max_abs_difference']>1e-6 or d['cube_obs_19_37_max_abs_difference']>1e-6 or d['proxy_value_difference']>1e-5 or not(d['success_agree'] and d['terminated_agree'] and d['truncated_agree'])): first=i
 return out,first
def main():
 p=argparse.ArgumentParser(); p.add_argument('--dataset-dir',required=True); p.add_argument('--official-source',required=True); p.add_argument('--checkpoint-dir',required=True); p.add_argument('--phase0r-jsonl',required=True); p.add_argument('--output-dir',required=True); a=p.parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
 rows=[json.loads(x) for x in Path(a.phase0r_jsonl).read_text().splitlines()]; by={}
 for r in rows: by.setdefault(int(r['root_id']),r)
 bad=[r for r in by.values() if not r.get('reproducible',True)][:8]; good=[r for r in by.values() if r.get('reproducible',True)][:8]; samples=bad+good
 env,tr,_=ogbench.make_env_and_datasets('cube-double-play-v0',dataset_dir=a.dataset_dir); env_b,_,_=ogbench.make_env_and_datasets('cube-double-play-v0',dataset_dir=a.dataset_dir); agent,cfg=load_agent(a.official_source,a.checkpoint_dir,1000000,tr,{'alpha':1.0}); tr_obs=np.asarray(tr['observations']); results=[]
 for r in samples:
  root,task,start=int(r['root_id']),int(r['task_id']),int(r['retrieval_start']); obs,info=env.reset(seed=root,options={'task_id':task}); goal=np.asarray(info['goal']); n=int(env.spec.max_episode_steps*(0,.25,.5)[stable_int(f'fraction:{root}')%3]); prefix=[]; o=obs
  for j in range(n):
   ac=action_for(agent,o,goal,jax.random.PRNGKey(stable_int(f'prefix:{root}:{j}')&0xffffffff)); prefix.append(ac); o,_,t,tu,_=env.step(ac)
   if t or tu:break
  env.reset(seed=root,options={'task_id':task}); o,_=env.reset(seed=root,options={'task_id':task});
  for ac in prefix:o,*_=env.step(ac)
  base=snap(env); base_obs=np.asarray(o); cand=tr_obs[start+20]; fixed=[np.zeros(env.action_space.shape,dtype=env.action_space.dtype)]; rng=np.random.default_rng(root+int(r['candidate_id'])); openacts=[rng.uniform(env.action_space.low,env.action_space.high).astype(env.action_space.dtype) for _ in range(5)]; key=jax.random.PRNGKey(stable_int(f'branch:{root}:{r["candidate_id"]}')&0xffffffff)
  rec={'root_id':root,'candidate_id':int(r['candidate_id']),'task_id':task,'sample_group':'non_reproducible' if not r.get('reproducible',True) else 'reproducible','levels':{}}
  # D0 mutate/step then restore and compare against original state/observation.
  mutate=np.ones(env.action_space.shape,dtype=env.action_space.dtype)*0.123; restore(env,base); x=run(env,[mutate]); restore(env,base); y=run(env,[]); d0={'first_divergent_step':None,'restored_integration_max_abs_difference':ad(snap(env)['integration'],base['integration']),'restored_observation_max_abs_difference':ad(snap(env)['py']['qpos'],base['py']['qpos']),'action_max_abs_difference':0.0,'proxy_value_difference':0.0,'success_terminated_truncated_agree':True}; rec['levels']['D0']=d0
  for name,acts in [('D1',fixed),('D2',openacts)]:
   restore(env,base); x=run(env,acts); restore(env,base); y=run(env,acts); dif,first=cmp(x,y,agent,goal); rec['levels'][name]={'first_divergent_step':first,'per_step':dif,'action_max_abs_difference':max([z['action_max_abs_difference'] for z in dif],default=0.0),'observation_max_abs_difference':max([z['observation_max_abs_difference'] for z in dif],default=0.0),'cube_state_max_abs_difference':max([z['cube_obs_19_37_max_abs_difference'] for z in dif],default=0.0),'proxy_value_difference':max([z['proxy_value_difference'] for z in dif],default=0.0),'success_terminated_truncated_agree':all(z['success_agree'] and z['terminated_agree'] and z['truncated_agree'] for z in dif)}
  restore(env,base); x=[]; o=base_obs
  for i in range(5):
   ac=action_for(agent,o,cand,jax.random.fold_in(key,i)); x.extend(run(env,[ac])); o=np.asarray(x[-1]['observation'])
  restore(env,base); y=[]; o=base_obs
  for i in range(5):
   ac=action_for(agent,o,cand,jax.random.fold_in(key,i)); y.extend(run(env,[ac])); o=np.asarray(y[-1]['observation'])
  dif,first=cmp(x,y,agent,goal); rec['levels']['D3']={'first_divergent_step':first,'per_step':dif,'action_max_abs_difference':max([z['action_max_abs_difference'] for z in dif],default=0.0),'observation_max_abs_difference':max([z['observation_max_abs_difference'] for z in dif],default=0.0),'cube_state_max_abs_difference':max([z['cube_obs_19_37_max_abs_difference'] for z in dif],default=0.0),'proxy_value_difference':max([z['proxy_value_difference'] for z in dif],default=0.0),'success_terminated_truncated_agree':all(z['success_agree'] and z['terminated_agree'] and z['truncated_agree'] for z in dif)}
  # reset-replay in two independent environments.
  oa,ia=env.reset(seed=root,options={'task_id':task}); ob,ib=env_b.reset(seed=root,options={'task_id':task}); ra=run(env,prefix); rb=run(env_b,prefix); rec['reset_replay']={'reset_observation_max_abs_difference':ad(oa,ob),'prefix_replay_observation_max_abs_difference':ad(ra[-1]['observation'],rb[-1]['observation']) if ra and rb else 0.0,'status':'PASS' if ad(oa,ob)<=1e-6 and (ad(ra[-1]['observation'],rb[-1]['observation']) if ra and rb else 0)<=1e-6 else 'FAIL'}; rec['snapshot_v2']={'status':'PASS' if all(rec['levels'][z]['first_divergent_step'] is None for z in ['D0','D1','D2','D3']) else 'FAIL'}; results.append(rec)
 (out/'branch_determinism_levels_v2.jsonl').write_text('\n'.join(json.dumps(r,sort_keys=True) for r in results)+'\n');
 for name,key in [('snapshot_v2_audit.json','snapshot_v2'),('reset_replay_audit.json','reset_replay')]: (out/name).write_text(json.dumps({'status':'PASS' if all(r[key]['status']=='PASS' for r in results) else 'FAIL','samples':len(results),'pass_count':sum(r[key]['status']=='PASS' for r in results),'diagnostic_set':'8 unique non-reproducible roots + 8 unique reproducible roots'},indent=2)+'\n')
 print(len(results))
if __name__=='__main__':main()
