import argparse,csv,hashlib,json,time,copy
from pathlib import Path
import jax,jax.numpy as jnp,numpy as np,ogbench
from execution_aligned_rl.evaluation.evaluate_gap import load_agent,action_for,value_for,stable_int,rank_corr
from execution_aligned_rl.data.audit_assets import episode_bounds,sha256_file
from execution_aligned_rl.evaluation.snapshot_v2 import capture_snapshot_v2,restore_snapshot_v2
def jd(x):
 return x.item() if isinstance(x,np.generic) else x
def run(env,agent,goal,target,key,gamma):
 o=env.unwrapped.compute_observation() if hasattr(env.unwrapped,'compute_observation') else env.unwrapped.get_ob(); rr=[]; acts=[]; succ=0.; term=tr=False
 for i in range(target):
  a=action_for(agent,o,goal,jax.random.fold_in(key,i)); acts.append(a); o,r,term,tr,inf=env.step(a); rr.append(float(r)-1.0); succ=max(succ,float(inf.get('success',0)))
  if term or tr: break
 return {'observation':np.asarray(o),'actions':acts,'discounted_reward':float(sum(gamma**i*x for i,x in enumerate(rr))),'return':float(sum(x+1 for x in rr)),'success':succ,'terminated':bool(term),'truncated':bool(tr),'steps':len(rr)}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--dataset-dir',required=True); p.add_argument('--official-source',required=True); p.add_argument('--checkpoint-dir',required=True); p.add_argument('--candidate-checkpoint',required=True); p.add_argument('--output-dir',required=True); a=p.parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
 env,tr,_=ogbench.make_env_and_datasets('cube-double-play-v0',dataset_dir=a.dataset_dir); agent,cfg=load_agent(a.official_source,a.checkpoint_dir,1000000,tr,{'alpha':1.0}); obs=np.asarray(tr['observations'],np.float32); starts=[]
 for s,e in episode_bounds(np.asarray(tr['terminals'])): starts.extend(range(s,e-20))
 starts=np.asarray(starts); scale=np.maximum(obs.std(0),1e-6); idx=np.arange(4000,4032); deep=set(sorted(idx,key=lambda r:stable_int(f'deep:{r}'))[:16]); raw=[]; roots=[]; t0=time.time(); tol=.5495205402374268
 for root in idx:
  task=root%5+1; frac=(0,.25,.5)[stable_int(f'fraction:{root}')%3]; pre=int(env.spec.max_episode_steps*frac); o,info=env.reset(seed=int(root),options={'task_id':task}); goal=np.asarray(info['goal']); done=False
  for j in range(pre):
   o,_,te,tu,_=env.step(action_for(agent,o,goal,jax.random.PRNGKey(stable_int(f'prefix:{root}:{j}')&0xffffffff)))
   if te or tu: done=True; break
  if done: roots.append({'root_id':int(root),'task_id':task,'legal':False,'reason':'prefix_terminated'}); continue
  snap=capture_snapshot_v2(env); q=np.asarray(o)/scale; dist=np.sqrt(np.mean(((obs[starts]/scale)-q)**2,axis=1)); order=np.argsort(dist); chosen=[]; seen=set()
  for ix in order:
   h=hashlib.sha256(obs[starts[ix]+20].tobytes()).hexdigest()
   if h not in seen: chosen.append((int(starts[ix]),obs[starts[ix]+20])); seen.add(h)
   if len(chosen)==8: break
  vals=value_for(agent,np.stack([x[1] for x in chosen]),goal); cr=[]
  for cid,(si,cand) in enumerate(chosen):
   reps=[]
   for rep in range(2):
    restore_snapshot_v2(env,snap); z=run(env,agent,cand,5,jax.random.PRNGKey(stable_int(f'branch:{root}:{cid}')&0xffffffff),float(cfg.discount)); z['proxy']=z['discounted_reward']+(float(cfg.discount)**z['steps'])*float(value_for(agent,z['observation'][None],goal)[0]); reps.append(z)
   same=bool(np.allclose(reps[0]['observation'],reps[1]['observation'],atol=1e-6) and abs(reps[0]['proxy']-reps[1]['proxy'])<=1e-5 and reps[0]['success']==reps[1]['success'] and reps[0]['terminated']==reps[1]['terminated'] and reps[0]['truncated']==reps[1]['truncated'])
   row={'root_id':int(root),'task_id':task,'candidate_id':cid,'retrieval_start':si,'ideal_value':float(vals[cid]),'repeats':[{'endpoint':r['observation'].tolist(),'proxy':r['proxy'],'success':r['success'],'terminated':r['terminated'],'truncated':r['truncated'],'return':r['return'],'steps':r['steps']} for r in reps],'reproducible':same,'result_source':'ENV_EVALUATED','training_eligible':False}; cr.append(row); raw.append(row)
  ideal=max(cr,key=lambda r:r['ideal_value']); proxy=max(cr,key=lambda r:r['repeats'][0]['proxy']); sm={'root_id':int(root),'task_id':task,'legal':True,'ideal_proxy_spearman':rank_corr([r['ideal_value'] for r in cr],[r['repeats'][0]['proxy'] for r in cr]),'proxy_regret':proxy['repeats'][0]['proxy']-ideal['repeats'][0]['proxy'],'proxy_wrong_beyond_tolerance':proxy['repeats'][0]['proxy']-ideal['repeats'][0]['proxy']>tol,'all_reproducible':all(r['reproducible'] for r in cr)}
  if root in deep:
   full=[]
   for r in cr:
    restore_snapshot_v2(env,snap); first=run(env,agent,obs[r['retrieval_start']+20],5,jax.random.PRNGKey(stable_int(f'deep:{root}:{r["candidate_id"]}')&0xffffffff),float(cfg.discount));
    if first['terminated'] or first['truncated']: z=first
    else:
     tail=run(env,agent,goal,500-first['steps'],jax.random.PRNGKey(stable_int(f'deep-tail:{root}:{r["candidate_id"]}')&0xffffffff),float(cfg.discount)); z={'observation':tail['observation'],'return':first['return']+tail['return'],'success':max(first['success'],tail['success']),'terminated':tail['terminated'],'truncated':tail['truncated'],'steps':first['steps']+tail['steps']}
    full.append((r,z))
   it=next(z for z in full if z[0]['candidate_id']==ideal['candidate_id']); ft=max(full,key=lambda x:(x[1]['success'],x[1]['return'])); sm.update({'full_distinguishable':len({x[1]['success'] for x in full})>1,'full_gap_present':ft[1]['success']>it[1]['success'],'ideal_top_full_success':it[1]['success'],'best_full_success':ft[1]['success'],'ideal_full_return':it[1]['return'],'best_full_return':ft[1]['return'],'ideal_completion_steps':it[1]['steps'],'best_completion_steps':ft[1]['steps']})
  roots.append(sm)
 rp=out/'phase0r2_candidates.jsonl'; rp.write_text('\n'.join(json.dumps(r,sort_keys=True,default=jd) for r in raw)+'\n'); rootp=out/'phase0r2_roots.csv'; keys=sorted({k for r in roots for k in r});
 with rootp.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(roots)
 legal=[r for r in roots if r['legal']]; deeprows=[r for r in legal if 'full_gap_present' in r]; wrong=float(np.mean([r['proxy_wrong_beyond_tolerance'] for r in legal])); repro=all(r['all_reproducible'] for r in legal); dist=sum(r.get('full_distinguishable',False) for r in deeprows); gaps=sum(r.get('full_gap_present',False) for r in deeprows); status='HOLD_CUBE_BRANCH_REPRODUCIBILITY' if not repro else ('CUBE_GAP_PRESENT_ROBUST' if len(legal)>=24 and len(deeprows)>=12 and wrong>=.1 and dist>=4 and gaps>=2 else ('HOLD_CUBE_INSUFFICIENT_COVERAGE' if len(legal)<24 or len(deeprows)<12 else ('CUBE_GAP_PROXY_ONLY_ROBUST' if wrong>=.1 else 'CUBE_NO_USEFUL_GAP_ROBUST')))
 summary={'status':status,'planned_roots':32,'legal_roots':len(legal),'planned_deep_roots':16,'deep_roots_evaluated':len(deeprows),'value_std_data_real':10.990410804748535,'value_tolerance':tol,'proxy_wrong_selection_rate':wrong,'mean_ideal_proxy_spearman':float(np.mean([r['ideal_proxy_spearman'] for r in legal if r['ideal_proxy_spearman'] is not None])),'full_distinguishable_roots':dist,'full_gap_roots':gaps,'ideal_not_best_deep_roots':gaps,'all_reproducible':repro,'raw_sha256':sha256_file(rp),'roots_sha256':sha256_file(rootp),'elapsed_seconds':time.time()-t0}; (out/'phase0r2_summary.json').write_text(json.dumps(summary,indent=2,default=jd)+'\n'); print(json.dumps(summary,indent=2,default=jd))
if __name__=='__main__':main()
