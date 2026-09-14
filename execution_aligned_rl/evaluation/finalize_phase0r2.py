import csv,hashlib,json,statistics,zipfile
from pathlib import Path

def sha(p):
 h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()

def main(root):
 root=Path(root); raw=root/'phase0r2_candidates.jsonl'; roots=root/'phase0r2_roots.csv'; summary=json.loads((root/'phase0r2_summary.json').read_text())
 rows=[json.loads(x) for x in raw.read_text().splitlines() if x.strip()]
 diffs=[]
 for r in rows:
  a,b=r['repeats']; diffs.append({'root_id':r['root_id'],'candidate_id':r['candidate_id'],'endpoint_max_abs':max(abs(x-y) for x,y in zip(a['endpoint'],b['endpoint'])),'proxy_abs':abs(a['proxy']-b['proxy']),'success_agree':a['success']==b['success'],'terminated_agree':a['terminated']==b['terminated'],'truncated_agree':a['truncated']==b['truncated'],'reproducible':r['reproducible']})
 audit={'protocol':'snapshot_v2','repeat_count':2,'candidate_branches':len(rows),'thresholds':{'endpoint_max_abs':1e-6,'proxy_abs':1e-5,'success_terminated_truncated_agreement':1.0},'max_endpoint_abs':max(d['endpoint_max_abs'] for d in diffs),'max_proxy_abs':max(d['proxy_abs'] for d in diffs),'agreement_rates':{k:sum(d[k] for d in diffs)/len(diffs) for k in ('success_agree','terminated_agree','truncated_agree')},'all_reproducible':summary['all_reproducible'],'rows':diffs}
 (root/'branch_reproducibility_r2.json').write_text(json.dumps(audit,indent=2)+'\n')
 protocol={'protocol_id':'cube-phase0r2','status':'frozen','ogbench_commit':'1d4140997f60c52c6fb0702ec100dc988b18c548','dataset':'cube-double-play-v0','train_sha256':'a73d1a33d029cedb8bc170ef94791ec585fa2d9450096f4f2a02b8cfbcf608c9','val_sha256':'b1fcdf4bd40750351a58d0d491d6be198366ce898f0c6a2e4cb5db331966013e','backbone_checkpoint':'/home/__compress_data/xushijie/og_runs/ea_v2_cube_mechanism_v1/checkpoints/gciql_seed0/params_1000000.pkl','candidate_source':'DATA_REAL train legal k=20 windows','roots':list(range(4000,4032)),'planned_roots':32,'planned_deep_roots':16,'k':20,'m':5,'N':8,'value_std':10.990410804748535,'value_tolerance':0.5495205402374268,'snapshot_interface':'execution_aligned_rl.evaluation.snapshot_v2','environment_source':'ENV_EVALUATED only','training_data_reentry':False,'gate':{'legal_roots_min':24,'deep_roots_min':12,'proxy_wrong_selection_min':0.1,'full_distinguishable_min':4,'ideal_not_best_deep_min':2}}
 (root/'frozen_phase0r2_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
 retrieval={'candidate_source':'DATA_REAL','window_horizon':20,'distinct_endpoints_per_root':8,'root_count':summary['legal_roots'],'candidate_count':len(rows),'training_eligible':False,'endpoint_validity':'train observations retrieved from legal same-episode windows','raw_candidates_sha256':sha(raw)}
 (root/'retrieval_candidate_audit_r2.json').write_text(json.dumps(retrieval,indent=2)+'\n')
 decision={'status':summary['status'],'phase_c_unlocked':False,'engineering_gate':{'legal_roots':summary['legal_roots'],'deep_roots':summary['deep_roots_evaluated'],'all_reproducible':summary['all_reproducible']},'mechanism_metrics':{'proxy_wrong_selection_rate':summary['proxy_wrong_selection_rate'],'full_distinguishable_roots':summary['full_distinguishable_roots'],'ideal_not_best_deep_roots':summary['ideal_not_best_deep_roots']},'reason':'Engineering reproducibility gate failed; no research conclusion computed.'}
 (root/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
 report=f'''# CUBE PHASE 0-R2 REPORT\n\nStatus: **{summary["status"]}**\n\nR2 used frozen Cube GCIQL/V assets, DATA_REAL legal k=20 train windows, roots 4000-4031, k=20, m=5, N=8, and the shared MuJoCo `snapshot_v2` integration-state interface. Candidate branches were evaluation-only (`ENV_EVALUATED`) and no result entered training data.\n\n## Coverage\n\n- Legal roots: {summary["legal_roots"]}/32\n- Deep roots evaluated: {summary["deep_roots_evaluated"]}/16\n- Proxy wrong-selection rate: {summary["proxy_wrong_selection_rate"]:.4f}\n- Full distinguishable roots: {summary["full_distinguishable_roots"]}\n- IDEAL not best deep roots: {summary["ideal_not_best_deep_roots"]}\n\n## Engineering Gate\n\nRepeated branches were compared from restored `mjSTATE_INTEGRATION` snapshots with Python mutable state, RNG, task state, and wrapper elapsed state restored. `all_reproducible` was **{summary["all_reproducible"]}**, so the mandatory engineering gate failed. Per protocol, the numerical mechanism thresholds are not interpreted as a research result and Phase C remains locked.\n'''
 (root/'CUBE_PHASE0R2_REPORT.md').write_text(report)
 files=[root/n for n in ('frozen_phase0r2_protocol.json','retrieval_candidate_audit_r2.json','phase0r2_candidates.jsonl','phase0r2_roots.csv','phase0r2_summary.json','branch_reproducibility_r2.json','CUBE_PHASE0R2_REPORT.md','decision.json')]
 z=root/'cube_phase0r2_lightweight.zip'
 with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as f:
  for p in files:f.write(p,p.name)
 (root/'cube_phase0r2_lightweight.zip.sha256').write_text(sha(z)+'  '+z.name+'\n')
if __name__=='__main__': main(__import__('sys').argv[1])
