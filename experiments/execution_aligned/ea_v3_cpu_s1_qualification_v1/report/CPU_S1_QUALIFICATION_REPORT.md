# EA-V3 CPU S1 Qualification Report

Status: `EA33_CPU_S1_QUALIFIED`

{
  "regression": {
    "legal": 14,
    "planned": 15,
    "tasks": [
      1,
      2,
      3,
      4,
      5
    ],
    "decision_steps_present": [
      0,
      125,
      250
    ],
    "ok": true,
    "ineligible": [
      710005
    ]
  },
  "holdout": {
    "legal": 13,
    "planned": 15,
    "tasks": [
      1,
      2,
      3,
      4,
      5
    ],
    "decision_steps_present": [
      0,
      125,
      250
    ],
    "ok": true,
    "ineligible": [
      711001,
      711005
    ]
  },
  "external_control_steps": 11316,
  "negatives": "PASS",
  "aba_ok": true,
  "identity_ok": true,
  "identities": {
    "r0_collector": "ce7dad6d2f2a9d8ec7dde9faa5ef6477b2cbd60da0a86c8ab9785549c24cf9e8",
    "r0_worker_A": "ce7dad6d2f2a9d8ec7dde9faa5ef6477b2cbd60da0a86c8ab9785549c24cf9e8",
    "r0_worker_B": "ce7dad6d2f2a9d8ec7dde9faa5ef6477b2cbd60da0a86c8ab9785549c24cf9e8",
    "r0_finalizer": "ce7dad6d2f2a9d8ec7dde9faa5ef6477b2cbd60da0a86c8ab9785549c24cf9e8",
    "collector": "ce7dad6d2f2a9d8ec7dde9faa5ef6477b2cbd60da0a86c8ab9785549c24cf9e8",
    "worker_A": "ce7dad6d2f2a9d8ec7dde9faa5ef6477b2cbd60da0a86c8ab9785549c24cf9e8",
    "worker_B": "ce7dad6d2f2a9d8ec7dde9faa5ef6477b2cbd60da0a86c8ab9785549c24cf9e8",
    "finalizer": "ce7dad6d2f2a9d8ec7dde9faa5ef6477b2cbd60da0a86c8ab9785549c24cf9e8"
  }
}

## Per-root

- regression 710000 task 1 step 0: PASS
- regression 710001 task 1 step 125: PASS
- regression 710002 task 1 step 250: PASS
- regression 710003 task 2 step 0: PASS
- regression 710004 task 2 step 125: PASS
- regression 710006 task 3 step 0: PASS
- regression 710007 task 3 step 125: PASS
- regression 710008 task 3 step 250: PASS
- regression 710009 task 4 step 0: PASS
- regression 710010 task 4 step 125: PASS
- regression 710011 task 4 step 250: PASS
- regression 710012 task 5 step 0: PASS
- regression 710013 task 5 step 125: PASS
- regression 710014 task 5 step 250: PASS
- holdout 711000 task 1 step 0: PASS
- holdout 711002 task 1 step 250: PASS
- holdout 711003 task 2 step 0: PASS
- holdout 711004 task 2 step 125: PASS
- holdout 711006 task 3 step 0: PASS
- holdout 711007 task 3 step 125: PASS
- holdout 711008 task 3 step 250: PASS
- holdout 711009 task 4 step 0: PASS
- holdout 711010 task 4 step 125: PASS
- holdout 711011 task 4 step 250: PASS
- holdout 711012 task 5 step 0: PASS
- holdout 711013 task 5 step 125: PASS
- holdout 711014 task 5 step 250: PASS
