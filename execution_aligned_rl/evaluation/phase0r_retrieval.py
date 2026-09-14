import argparse,csv,hashlib,json,pickle,time
from pathlib import Path
import jax,numpy as np,ogbench
from execution_aligned_rl.evaluation.evaluate_gap import load_agent,action_for,value_for,run_steps,stable_int,rank_corr
from execution_aligned_rl.data.audit_assets import capture_snapshot,restore_snapshot,episode_bounds,sha256_file

def main():
 p=argparse.ArgumentParser(); p.add_argument('--dataset-dir',required=True); p.add_argument('--official-source',required=True); p.add_argument('--checkpoint-dir',required=True); p.add_argument('--candidate-checkpoint',required=True); p.add_argument('--output-dir',required=True); p.add_argument('--root-offset',type=int,default=3000); p.add_argument('--roots',type=int,default=32); p.add_argument('--deep-roots',type=int,default=16); p.add_argument('--k',type=int,default=20); p.add_argument('--m',type=int,default=5); a=p.parse_args()
 out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); env,train,_=ogbench.make_env_and_datasets('cube-double-play-v0',dataset_dir=a.dataset_dir); agent,cfg=load_agent(a.official_source,a.checkpoint_dir,1000000,train,{'alpha':1.0}); tr=np.asarray(train['observations'],np.float32); terms=np.asarray(train['terminals']); starts=[]
 for s,e in episode_bounds(terms): starts.extend(range(s,e-a.k))
 starts=np.asarray(starts); scale=np.maximum(tr.std(0),1e-6); norm_starts=tr[starts]/scale; roots=list(range(a.root_offset,a.root_offset+a.roots)); deep=set(sorted(roots,key=lambda r:stable_int(f'deep:{r}'))[:a.deep_roots]); raw=[]; summaries=[]; started=time.time()
 for root in roots:
  task=root%len(env.unwrapped.task_infos)+1; frac=(0,.25,.5)[stable_int(f'fraction:{root}')%3]; ds=int(env.spec.max_episode_steps*frac); obs,info=env.reset(seed=root,options={'task_id':task}); goal=np.asarray(info['goal']); bad=False
  for j in range(ds):
   obs,_,te,trunc,_=env.step(action_for(agent,obs,goal,jax.random.PRNGKey(stable_int(f'prefix:{root}:{j}')&0xffffffff)))
   if te or trunc: bad=True; break
  if bad: summaries.append({'root_id':root,'task_id':task,'legal':False,'reason':'prefix_terminated'}); continue
  snap=capture_snapshot(env); q=np.asarray(obs)/scale; d=np.sqrt(np.mean((norm_starts-q)**2,axis=1)); order=np.argsort(d); chosen=[]; seen=set()
  for ix in order:
   ep=tr[starts[ix]+a.k]
   h=hashlib.sha256(ep.tobytes()).hexdigest()
   if h not in seen: chosen.append((ix,ep)); seen.add(h)
   if len(chosen)>=8: break
  vals=value_for(agent,np.stack([x[1] for x in chosen]),goal); rows=[]
  for cid,(ix,cand) in enumerate(chosen):
   key=jax.random.PRNGKey(stable_int(f'branch:{root}:{cid}')&0xffffffff); reps=[]
   for rep in range(2):
    restore_snapshot(env,snap); first=run_steps(env,agent,cand,key,a.m,float(cfg.discount)); tail=0 if first['terminated'] or first['truncated'] else float(value_for(agent,first['observation'][None],goal)[0]); proxy=first['discounted_training_reward']+(float(cfg.discount)**first['steps'])*tail; reps.append({'m_endpoint':first['observation'].tolist(),'proxy_value':proxy,'success':first['success'],'terminated':first['terminated'],'truncated':first['truncated'],'pose_distance':float(np.linalg.norm(first['observation'][12:]-reps[0]['m_endpoint'][12:])) if rep else 0.0})
   reproducible=all(np.allclose(np.asarray(reps[0]['m_endpoint']),np.asarray(reps[1]['m_endpoint']),atol=1e-7) and abs(reps[0]['proxy_value']-reps[1]['proxy_value'])<1e-7 and reps[0]['success']==reps[1]['success'] and reps[0]['terminated']==reps[1]['terminated'] and reps[0]['truncated']==reps[1]['truncated'] for _ in [0])
   row={'root_id':root,'task_id':task,'candidate_id':cid,'k':a.k,'m':a.m,'ideal_value':float(vals[cid]),'retrieval_start':int(starts[ix]),'retrieval_nn_distance':float(d[ix]),'repeat_1':reps[0],'repeat_2':reps[1],'reproducible':reproducible,'result_source':'ENV_EVALUATED','training_eligible':False}; rows.append(row); raw.append(row)
  ideal=max(rows,key=lambda r:r['ideal_value']); proxy=max(rows,key=lambda r:r['repeat_1']['proxy_value']); sm={'root_id':root,'task_id':task,'legal':True,'ideal_proxy_spearman':rank_corr([r['ideal_value'] for r in rows],[r['repeat_1']['proxy_value'] for r in rows]),'proxy_regret':proxy['repeat_1']['proxy_value']-ideal['repeat_1']['proxy_value'],'proxy_wrong_beyond_tolerance':proxy['repeat_1']['proxy_value']-ideal['repeat_1']['proxy_value']>0.05*np.std(vals),'all_reproducible':all(r['reproducible'] for r in rows)}
  if root in deep:
   full=[]
   for r in rows:
    restore_snapshot(env,snap); key=jax.random.PRNGKey(stable_int(f'deep:{root}:{r["candidate_id"]}')&0xffffffff); z=run_steps(env,agent,tr[r['retrieval_start']+a.k],key,40,float(cfg.discount)); full.append((r,z))
   ft=max(full,key=lambda x:(x[1]['success'],x[1]['environment_reward_sum'])); it=next(z[1] for z in full if z[0]['candidate_id']==ideal['candidate_id']); sm.update({'full_distinguishable':len({x[1]['success'] for x in full})>1,'full_gap_present':ft[1]['success']>it['success'],'ideal_top_full_success':it['success'],'best_full_success':ft[1]['success']})
  summaries.append(sm)
 rawp=out/'phase0r_candidates.jsonl'; rawp.write_text('\n'.join(json.dumps(r,sort_keys=True) for r in raw)+'\n'); rootp=out/'phase0r_roots.csv'; keys=sorted({k for r in summaries for k in r});
 with rootp.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(summaries)
 legal=[r for r in summaries if r['legal']]; deeprows=[r for r in legal if 'full_gap_present' in r]; wrong=float(np.mean([r['proxy_wrong_beyond_tolerance'] for r in legal])); dist=sum(not r['full_distinguishable'] is False for r in deeprows); gaps=sum(r['full_gap_present'] for r in deeprows); summary={'status':'CUBE_GAP_PRESENT_ROBUST' if wrong>=.1 and dist>=4 and gaps>=2 else 'CUBE_GAP_PROXY_ONLY_ROBUST' if wrong>=.1 else 'CUBE_NO_USEFUL_GAP_ROBUST','planned_roots':a.roots,'legal_roots':len(legal),'planned_deep_roots':a.deep_roots,'deep_roots_evaluated':len(deeprows),'full_distinguishable_roots':sum(r.get('full_distinguishable',False) for r in deeprows),'full_gap_roots':gaps,'ideal_not_best_deep_roots':gaps,'proxy_wrong_selection_rate':wrong,'mean_ideal_proxy_spearman':float(np.mean([r['ideal_proxy_spearman'] for r in legal if r['ideal_proxy_spearman'] is not None])),'all_reproducible':all(r['all_reproducible'] for r in legal),'raw_sha256':sha256_file(rawp),'roots_sha256':sha256_file(rootp),'elapsed_seconds':time.time()-started}; (out/'phase0r_summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
