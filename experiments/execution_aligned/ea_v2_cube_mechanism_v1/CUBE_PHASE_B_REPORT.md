# Cube Phase B Report

Status: `CUBE_PHASE_B_PASS_PROVISIONAL`.

Official Cube GCIQL used alpha `1.0`, actor goal defaults randomgoal `0.0` / trajgoal `1.0`, seed `0`, 1,000,000 updates, and fixed-final checkpoint. Overall 250-episode success was `0.36`; task rates were `0.50, 0.52, 0.58, 0.08, 0.12`. All five tasks had nonzero success.

Action bounds, finite value rate, candidate finite/distinct rates, and complete 37D candidate-to-GCIQL goal compatibility passed. The pinned evaluator did not emit separate return or episode-length aggregates; these are recorded as unavailable without inference. No Phase 0 or later phase was run.
