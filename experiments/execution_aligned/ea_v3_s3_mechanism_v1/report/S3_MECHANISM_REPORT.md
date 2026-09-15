# S3 Mechanism Report

Status: `EA35_GAP_EXISTS_PROXY_MISALIGNED`

- research_metrics_valid: `True`
- s4_unlocked: `false`
- training_performed: `false`
- runtime identity: `ce7dad6d2f2a9d8ec7dde9faa5ef6477b2cbd60da0a86c8ab9785549c24cf9e8`
- external control steps: `412726` / `800000`

## Engineering

{
  "holds": [],
  "n_failures": 0
}

## Science

{
  "protocol_id": "ea_v3_s3_mechanism_v1",
  "research_metrics_valid": true,
  "n_legal_roots": 70,
  "n_legal_deep_roots": 35,
  "wrong_selection_count": 10,
  "wrong_selection_rate": 0.14285714285714285,
  "full_distinguishable": 17,
  "ideal_full_miss": 6,
  "proxy_paired_net_success_gain": -5,
  "ideal_full_success_rate": 0.42857142857142855,
  "proxy_full_success_rate": 0.2857142857142857,
  "oracle_full_success_rate": 0.6,
  "direct_success_rate": 0.37142857142857144,
  "uniform_expectation": 0.32857142857142857,
  "mean_spearman_proxy_vs_success": 0.02098391320912746,
  "ideal_value_ties": 0,
  "proxy_ties": 0,
  "gates": {
    "wrong_selection_min_count": 7,
    "distinguishable_min": 9,
    "ideal_miss_min": 5,
    "paired_net_gain_min": 0,
    "wrong_selection_count": 10,
    "wrong_selection_rate": 0.14285714285714285,
    "distinguishable": 17,
    "ideal_miss": 6,
    "paired_net_gain": -5,
    "wrong_ok": true,
    "dist_ok": true,
    "miss_ok": true,
    "gain_ok": false
  },
  "engineering_holds": []
}

## Bootstrap 95% CI

{
  "repetitions": 10000,
  "seed": 350301,
  "unit": "root",
  "stratified_by_task": true,
  "wrong_selection_rate": {
    "mean": 0.14280714285714285,
    "lo": 0.07142857142857142,
    "hi": 0.22857142857142856
  },
  "distinguishable": {
    "mean": 17.0159,
    "lo": 11.0,
    "hi": 24.0
  },
  "ideal_miss": {
    "mean": 5.9689,
    "lo": 2.0,
    "hi": 11.0
  },
  "paired_net_gain": {
    "mean": -5.0025,
    "lo": -9.0,
    "hi": -1.0
  }
}

