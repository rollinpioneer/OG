# EA-V4 Outcome Evaluator Pilot Report

Status: `EA41_EVALUATOR_NO_SIGNAL`

This is an exploratory offline evaluator pilot, not a preregistered confirmation.

## Locks

- new_environment_steps: 0
- fresh_confirmation_unlocked: false
- new_environment_evaluation_authorized: false
- policy_training_authorized: false
- s4_unlocked: false
- Gate A: EA40_SUPPORT_AND_DATA_PASS
- leakage: PASS
- training seal: TRAINING_SEALED

## References

- IDEAL 15/35
- ORACLE_FULL 21/35
- ORACLE_PROXY 10/35
- DIRECT_35 14/35

## Ensemble

- B0: success=15/35 rescue=0 harm=0 net=0 interv=0 pairwise_auc=0.5489145658263306
- B1: success=9/35 rescue=2 harm=8 net=-6 interv=26 pairwise_auc=0.4566176470588235
- B2: success=9/35 rescue=2 harm=8 net=-6 interv=31 pairwise_auc=0.4688725490196079
- B3: success=12/35 rescue=2 harm=5 net=-3 interv=23 pairwise_auc=0.5047268907563025
- B4: success=11/35 rescue=2 harm=6 net=-4 interv=26 pairwise_auc=0.46302521008403363
- B5: success=15/35 rescue=1 harm=1 net=0 interv=6 pairwise_auc=0.46302521008403363

## Bootstrap (task-stratified, 10000)

```json
{
  "B0": {
    "harm": {
      "mean": 0.0,
      "p025": 0.0,
      "p975": 0.0
    },
    "net": {
      "mean": 0.0,
      "p025": 0.0,
      "p975": 0.0
    },
    "rescue": {
      "mean": 0.0,
      "p025": 0.0,
      "p975": 0.0
    },
    "success": {
      "mean": 15.0251,
      "p025": 10.0,
      "p975": 20.0
    }
  },
  "B2": {
    "harm": {
      "mean": 7.9759,
      "p025": 4.0,
      "p975": 12.0
    },
    "net": {
      "mean": -5.9564,
      "p025": -11.0,
      "p975": 0.0
    },
    "rescue": {
      "mean": 2.0195,
      "p025": 0.0,
      "p975": 5.0
    },
    "success": {
      "mean": 8.9958,
      "p025": 5.0,
      "p975": 14.0
    }
  },
  "B4": {
    "harm": {
      "mean": 6.0071,
      "p025": 2.0,
      "p975": 10.0
    },
    "net": {
      "mean": -3.9979,
      "p025": -9.0,
      "p975": 1.0
    },
    "rescue": {
      "mean": 2.0092,
      "p025": 0.0,
      "p975": 5.0
    },
    "success": {
      "mean": 10.9946,
      "p025": 7.0,
      "p975": 15.024999999999636
    }
  },
  "B5": {
    "harm": {
      "mean": 0.9995,
      "p025": 0.0,
      "p975": 3.0
    },
    "net": {
      "mean": 0.0099,
      "p025": -3.0,
      "p975": 3.0
    },
    "rescue": {
      "mean": 1.0094,
      "p025": 0.0,
      "p975": 3.0
    },
    "success": {
      "mean": 15.0358,
      "p025": 10.0,
      "p975": 20.0
    }
  }
}
```

## Decision locks

```json
{
  "fresh_confirmation_unlocked": false,
  "human_review": null,
  "new_environment_evaluation_authorized": false,
  "new_environment_steps": 0,
  "policy_training_authorized": false,
  "s4_unlocked": false,
  "status": "EA41_EVALUATOR_NO_SIGNAL"
}
```
