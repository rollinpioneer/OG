# S3 Postmortem Report

This is an **exploratory postmortem**, not a preregistered confirmation trial.
S3 sealed status remains `EA35_GAP_EXISTS_PROXY_MISALIGNED`. This analysis does not modify S3.

Postmortem status: `EA35X_POSTMORTEM_COMPLETE`

## P1 reproduction

{
  "n_legal": 70,
  "n_deep": 35,
  "wrong_selection": 10,
  "distinguishable": 17,
  "ideal_miss": 6,
  "paired_net_gain": -5,
  "ideal_success_35": 15,
  "proxy_success_35": 10,
  "full_success_35": 21,
  "direct_success_70": 26,
  "direct_success_35": 14,
  "uniform_35": 0.32857142857142857
}

Rescue/harm vs IDEAL on the same 35 deep roots:
{
  "ORACLE_PROXY": {
    "n_rescue": 0,
    "n_harm": 5,
    "n_retained_success": 10,
    "n_retained_failure": 20,
    "net": -5,
    "n_intervened_success_change": 5
  },
  "ORACLE_FULL": {
    "n_rescue": 6,
    "n_harm": 0,
    "n_retained_success": 15,
    "n_retained_failure": 14,
    "net": 6,
    "n_intervened_success_change": 6
  },
  "DIRECT_on_35": {
    "n_rescue": 4,
    "n_harm": 5,
    "n_retained_success": 10,
    "n_retained_failure": 16,
    "net": -1,
    "n_intervened_success_change": 9
  }
}

## P3 horizons

All six frozen horizons are reported. A longer h that has already observed success is not a deployable algorithm.
{
  "1": {
    "J_h": {
      "success": 11,
      "rate": 0.3142857142857143,
      "n_rescue": 0,
      "n_harm": 4,
      "n_retained_success": 11,
      "n_retained_failure": 20,
      "net": -4,
      "n_intervened_success_change": 4
    },
    "IDEAL": {
      "success": 15,
      "rate": 0.42857142857142855,
      "n_rescue": 0,
      "n_harm": 0,
      "n_retained_success": 15,
      "n_retained_failure": 20,
      "net": 0,
      "n_intervened_success_change": 0
    },
    "KNOWN_OUTCOME_ONLY": {
      "success": 15,
      "rate": 0.42857142857142855,
      "n_rescue": 0,
      "n_harm": 0,
      "n_retained_success": 15,
      "n_retained_failure": 20,
      "net": 0,
      "n_intervened_success_change": 0
    },
    "n_candidate_prefixes_revealed": 0,
    "n_candidate_prefixes": 280,
    "revealed_fraction": 0.0
  },
  "5": {
    "J_h": {
      "success": 10,
      "rate": 0.2857142857142857,
      "n_rescue": 0,
      "n_harm": 5,
      "n_retained_success": 10,
      "n_retained_failure": 20,
      "net": -5,
      "n_intervened_success_change": 5
    },
    "IDEAL": {
      "success": 15,
      "rate": 0.42857142857142855,
      "n_rescue": 0,
      "n_harm": 0,
      "n_retained_success": 15,
      "n_retained_failure": 20,
      "net": 0,
      "n_intervened_success_change": 0
    },
    "KNOWN_OUTCOME_ONLY": {
      "success": 15,
      "rate": 0.42857142857142855,
      "n_rescue": 0,
      "n_harm": 0,
      "n_retained_success": 15,
      "n_retained_failure": 20,
      "net": 0,
      "n_intervened_success_change": 0
    },
    "n_candidate_prefixes_revealed": 0,
    "n_candidate_prefixes": 280,
    "revealed_fraction": 0.0
  },
  "10": {
    "J_h": {
      "success": 12,
      "rate": 0.34285714285714286,
      "n_rescue": 1,
      "n_harm": 4,
      "n_retained_success": 11,
      "n_retained_failure": 19,
      "net": -3,
      "n_intervened_success_change": 5
    },
    "IDEAL": {
      "success": 15,
      "rate": 0.42857142857142855,
      "n_rescue": 0,
      "n_harm": 0,
      "n_retained_success": 15,
      "n_retained_failure": 20,
      "net": 0,
      "n_intervened_success_change": 0
    },
    "KNOWN_OUTCOME_ONLY": {
      "success": 15,
      "rate": 0.42857142857142855,
      "n_rescue": 0,
      "n_harm": 0,
      "n_retained_success": 15,
      "n_retained_failure": 20,
      "net": 0,
      "n_intervened_success_change": 0
    },
    "n_candidate_prefixes_revealed": 0,
    "n_candidate_prefixes": 280,
    "revealed_fraction": 0.0
  },
  "20": {
    "J_h": {
      "success": 7,
      "rate": 0.2,
      "n_rescue": 0,
      "n_harm": 8,
      "n_retained_success": 7,
      "n_retained_failure": 20,
      "net": -8,
      "n_intervened_success_change": 8
    },
    "IDEAL": {
      "success": 15,
      "rate": 0.42857142857142855,
      "n_rescue": 0,
      "n_harm": 0,
      "n_retained_success": 15,
      "n_retained_failure": 20,
      "net": 0,
      "n_intervened_success_change": 0
    },
    "KNOWN_OUTCOME_ONLY": {
      "success": 15,
      "rate": 0.42857142857142855,
      "n_rescue": 0,
      "n_harm": 0,
      "n_retained_success": 15,
      "n_retained_failure": 20,
      "net": 0,
      "n_intervened_success_change": 0
    },
    "n_candidate_prefixes_revealed": 5,
    "n_candidate_prefixes": 280,
    "revealed_fraction": 0.017857142857142856
  },
  "40": {
    "J_h": {
      "success": 15,
      "rate": 0.42857142857142855,
      "n_rescue": 3,
      "n_harm": 3,
      "n_retained_success": 12,
      "n_retained_failure": 17,
      "net": 0,
      "n_intervened_success_change": 6
    },
    "IDEAL": {
      "success": 15,
      "rate": 0.42857142857142855,
      "n_rescue": 0,
      "n_harm": 0,
      "n_retained_success": 15,
      "n_retained_failure": 20,
      "net": 0,
      "n_intervened_success_change": 0
    },
    "KNOWN_OUTCOME_ONLY": {
      "success": 15,
      "rate": 0.42857142857142855,
      "n_rescue": 0,
      "n_harm": 0,
      "n_retained_success": 15,
      "n_retained_failure": 20,
      "net": 0,
      "n_intervened_success_change": 0
    },
    "n_candidate_prefixes_revealed": 14,
    "n_candidate_prefixes": 280,
    "revealed_fraction": 0.05
  },
  "80": {
    "J_h": {
      "success": 16,
      "rate": 0.45714285714285713,
      "n_rescue": 6,
      "n_harm": 5,
      "n_retained_success": 10,
      "n_retained_failure": 14,
      "net": 1,
      "n_intervened_success_change": 11
    },
    "IDEAL": {
      "success": 15,
      "rate": 0.42857142857142855,
      "n_rescue": 0,
      "n_harm": 0,
      "n_retained_success": 15,
      "n_retained_failure": 20,
      "net": 0,
      "n_intervened_success_change": 0
    },
    "KNOWN_OUTCOME_ONLY": {
      "success": 15,
      "rate": 0.42857142857142855,
      "n_rescue": 0,
      "n_harm": 0,
      "n_retained_success": 15,
      "n_retained_failure": 20,
      "net": 0,
      "n_intervened_success_change": 0
    },
    "n_candidate_prefixes_revealed": 29,
    "n_candidate_prefixes": 280,
    "revealed_fraction": 0.10357142857142858
  }
}

## P4 epsilon-switch

{
  "epsilon": 0.5495205402374268,
  "n_intervened": 3,
  "intervention_rate": 0.08571428571428572,
  "always_IDEAL": {
    "success": 15,
    "n_rescue": 0,
    "n_harm": 0,
    "n_retained_success": 15,
    "n_retained_failure": 20,
    "net": 0,
    "n_intervened_success_change": 0
  },
  "unconditional_ORACLE_PROXY": {
    "success": 10,
    "n_rescue": 0,
    "n_harm": 5,
    "n_retained_success": 10,
    "n_retained_failure": 20,
    "net": -5,
    "n_intervened_success_change": 5
  },
  "epsilon_switch": {
    "success": 15,
    "n_rescue": 0,
    "n_harm": 0,
    "n_retained_success": 15,
    "n_retained_failure": 20,
    "net": 0,
    "n_intervened_success_change": 0
  },
  "intervened": [
    {
      "root_id": 720023,
      "gap": 0.6654906070858218,
      "from": 5,
      "to": 7
    },
    {
      "root_id": 720060,
      "gap": 0.6117456516824262,
      "from": 3,
      "to": 6
    },
    {
      "root_id": 720061,
      "gap": 0.6791490449591748,
      "from": 4,
      "to": 1
    }
  ],
  "exploratory": true,
  "does_not_modify_s3": true
}

## Evidence fields

{
  "target_policy_mismatch": {
    "value": "possible_not_identified_as_unique_cause",
    "numbers": {
      "value_is_expectile_not_FQE": true,
      "success_without_done": 0
    },
    "source": "ogbench impls/agents/gciql.py value_loss; S3 compute_proxy",
    "unidentified": [
      "whether replacing V with FQE would reverse the -5 net"
    ]
  },
  "longer_horizon_descriptive_signal": {
    "value": true,
    "numbers": {
      "1": 11,
      "5": 10,
      "10": 12,
      "20": 7,
      "40": 15,
      "80": 16
    },
    "source": "deep traces J_h",
    "unidentified": [
      "decision-time predictability"
    ]
  },
  "signal_beyond_revealed_outcomes": {
    "value": true,
    "numbers": {
      "unresolved_h5": {
        "n_roots": 35,
        "tasks": [
          1,
          2,
          3,
          4,
          5
        ],
        "coverage": "OK",
        "IDEAL": {
          "n": 35,
          "success": 15,
          "rate": 0.42857142857142855,
          "n_rescue": 0,
          "n_harm": 0,
          "n_retained_success": 15,
          "n_retained_failure": 20,
          "net": 0,
          "n_intervened_success_change": 0
        },
        "J_h": {
          "n": 35,
          "success": 10,
          "rate": 0.2857142857142857,
          "n_rescue": 0,
          "n_harm": 5,
          "n_retained_success": 10,
          "n_retained_failure": 20,
          "net": -5,
          "n_intervened_success_change": 5
        },
        "KNOWN_OUTCOME_ONLY": {
          "n": 35,
          "success": 15,
          "rate": 0.42857142857142855,
          "n_rescue": 0,
          "n_harm": 0,
          "n_retained_success": 15,
          "n_retained_failure": 20,
          "net": 0,
          "n_intervened_success_change": 0
        },
        "root_ids": [
          720000,
          720006,
          720012,
          720013,
          720015,
          720017,
          720021,
          720022,
          720023,
          720025,
          720031,
          720032,
          720033,
          720034,
          720038,
          720039,
          720042,
          720044,
          720046,
          720052,
          720054,
          720055,
          720057,
          720058,
          720060,
          720061,
          720062,
          720066,
          720067,
          720069,
          720075,
          720076,
          720077,
          720078,
          720079
        ]
      },
      "unresolved_h80_n": 30,
      "known_h80_success": 15,
      "J_h80_success": 16
    },
    "source": "ALL_UNRESOLVED_h and KNOWN_OUTCOME_ONLY_h",
    "unidentified": [
      "offline learnability of any residual signal"
    ]
  },
  "epsilon_switch_signal": {
    "value": false,
    "numbers": {
      "epsilon": 0.5495205402374268,
      "n_intervened": 3,
      "intervention_rate": 0.08571428571428572,
      "always_IDEAL": {
        "success": 15,
        "n_rescue": 0,
        "n_harm": 0,
        "n_retained_success": 15,
        "n_retained_failure": 20,
        "net": 0,
        "n_intervened_success_change": 0
      },
      "unconditional_ORACLE_PROXY": {
        "success": 10,
        "n_rescue": 0,
        "n_harm": 5,
        "n_retained_success": 10,
        "n_retained_failure": 20,
        "net": -5,
        "n_intervened_success_change": 5
      },
      "epsilon_switch": {
        "success": 15,
        "n_rescue": 0,
        "n_harm": 0,
        "n_retained_success": 15,
        "n_retained_failure": 20,
        "net": 0,
        "n_intervened_success_change": 0
      },
      "intervened": [
        {
          "root_id": 720023,
          "gap": 0.6654906070858218,
          "from": 5,
          "to": 7
        },
        {
          "root_id": 720060,
          "gap": 0.6117456516824262,
          "from": 3,
          "to": 6
        },
        {
          "root_id": 720061,
          "gap": 0.6791490449591748,
          "from": 4,
          "to": 1
        }
      ],
      "exploratory": true,
      "does_not_modify_s3": true
    },
    "source": "fixed epsilon 0.5495205402374268",
    "unidentified": [
      "whether a learned rejector would generalize"
    ]
  },
  "offline_supervision_feasibility": {
    "value": true,
    "numbers": {
      "n_steps": 90680,
      "agree": 90680,
      "disagree": 0,
      "agreement_rate": 1.0,
      "examples": []
    },
    "source": "cube_env._compute_successes 0.04 xyz; public obs cube xyz scaled by 10",
    "unidentified": [
      "sufficient target-actor coverage for FQE"
    ]
  }
}

