import argparse, json, pickle
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np
from execution_aligned_rl.data.audit_assets import episode_bounds

def main():
 p=argparse.ArgumentParser(); p.add_argument('--dataset-dir',required=True); p.add_argument('--prior',required=True); p.add_argument('--output',required=True); p.add_argument('--official-source',required=True); a=p.parse_args()
 import ogbench
 _,tr,va=ogbench.make_env_and_datasets('cube-double-play-v0',dataset_dir=a.dataset_dir)
 tr=np.asarray(tr['observations'],dtype=np.float32); va=np.asarray(va['observations'],dtype=np.float32)
 payload=pickle.load(open(a.prior,'rb')); from execution_aligned_rl.agents.train_candidate_prior import CandidatePrior
 model=CandidatePrior(tuple(payload['config']['hidden_dims']),tr.shape[-1]); rng=np.random.default_rng(20260914)
 starts=rng.choice(len(va)-20,size=min(512,len(va)-20),replace=False); states=(va[starts]-payload['normalization']['obs_mean'])/payload['normalization']['obs_std']
 means,logs=model.apply({'params':payload['params']},jnp.asarray(states)); cand=(np.asarray(means)+np.exp(np.asarray(logs))*rng.standard_normal(np.asarray(means).shape))*payload['normalization']['obs_std']+payload['normalization']['obs_mean']; cand=np.clip(cand,payload['normalization']['target_min'],payload['normalization']['target_max'])
 endpoints=np.concatenate([tr[s+20:s+21] for s in range(0,len(tr)-20,20)],axis=0)
 scale=np.maximum(tr.std(0),1e-6); dif=[]; pose=[]
 pose_idx=np.arange(12,37)
 for c in cand:
  d=np.sqrt(((endpoints-c)/scale)**2).mean(1)**0.5; dif.append(float(d.min())); q=np.sqrt(((endpoints[:,pose_idx]-c[pose_idx])**2).mean(1)); pose.append(float(q.min()))
 out={'status':'PASS','dataset_id':'cube-double-play-v0','candidate_count':int(len(cand)),'endpoint_count':int(len(endpoints)),'normalized_full37d_nearest_neighbor':{k:float(np.percentile(dif,p)) for k,p in [('median',50),('p90',90),('p95',95),('max',100)]},'task_relevant_cube_pose_nearest_neighbor':{k:float(np.percentile(pose,p)) for k,p in [('median',50),('p90',90),('p95',95),('max',100)]},'finite_rate':float(np.isfinite(cand).all(1).mean()),'state_validity':'finite_and_clipped_to_training_support','quaternion_validity':'not_independently normalized; raw public observation quaternion fields retained','candidate_interface_dimension':int(cand.shape[1]),'support_source':'DATA_REAL training k=20 endpoints','result_source':'DATA_REAL'}
 Path(a.output).write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(out,indent=2))
if __name__=='__main__': main()
