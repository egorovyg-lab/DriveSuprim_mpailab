# Continuous diagnostics on frozen DS/Drex choices

This adds evaluation axes without changing PDM weights, prompts, decisions or the
finished replacement experiment. It is an exploratory TRAIN audit, not a new
selector, calibrated safety assessment, full EPDMS or independent evaluation.

## Fixed scope and inputs

Default source: `replacement_gpu2_v1`, all 651 scenes / 32 sessions. Compare DS,
`none/both050`, `blank/both050`, `road/both050`, `blank/both075`, `road/both075`,
`old_table`, `old_camera`. Their union contains 1041 unique token/candidate pairs
in the supplied completed CSV. Selection never uses newly measured diagnostics.
The oracle is intentionally absent. The `blank` arms retain a matched control
for camera information. All unchanged scenes remain in aggregate comparisons.

Most original scoring calls stored only metric rows. This tool therefore repeats
the CPU simulator/scorer for the selected union, not the 246078 Drex requests or
all top64 proposals. Every replay must reproduce all nine original raw values
(`atol=1e-8, rtol=1e-7`) before new measurements are accepted. It follows existing
SHA-bound cache/config dependencies and reopens a fresh cache for each candidate.
Top64 tail labels come from the already verified external label source; they do
not have to exist in the original top16 NPZ. Scorer sources, cache bytes, local /
vocabulary IDs, vehicle dimensions, package, tools and environment are checked.

## Definitions

* Sampling: 41 states at 0.1 s over four seconds. Both the entire horizon and
  `[1,4]` seconds are reported: an identical initial clearance can dominate the
  full-horizon minimum. Removing the first second is not a safety claim.
* Road: union of the scorer's ROADBLOCK, INTERSECTION, DRIVABLE_AREA and CARPARK_AREA
  polygons. Use the **whole scorer footprint** and remove internal polygon seams.
  Minimum clearance is null if any sampled footprint is outside. Also report
  outside area maximum/integral, outside time and exposure below 0.1/0.25/0.5 m.
  Signed corner margins are separate; they do not describe the entire body.
* Other participants: minimum polygon distance and observed exposure below
  0.5/1/2 m. Use the original non-reactive traffic policy, including removal of
  initially overlapping objects. No actors means null distance, not zero or
  infinity. Observation interval coverage is explicit. Geometry contacts are not
  at-fault collisions, TTC probabilities or exact continuous swept collisions.
* Comfort: shift simulated rear-axle dynamic states to the **cached vehicle
  center** using the existing nuPlan helper. Differentiate world-frame center
  velocity into physical acceleration, then differentiate again into jerk.
  Rotate to the body frame for longitudinal/lateral components. This includes
  centripetal acceleration, which the bicycle simulator's stored lateral
  acceleration field omits. Angular velocity/acceleration remain state signals. RMS, absolute
  sample p95 and maximum are saved. No signal smoothing or threshold tuning.
  These values differ from the filtered/binary history-comfort calculation.
  Include moving-only summaries (speed >=0.5 m/s) and their time coverage so
  stopping is not silently equated with better driving. Sample RMS has trapezoid
  endpoint weights for state signals; acceleration uses 40 interval values and
  jerk uses 39 midpoint values (3.9 s support), with no fabricated endpoint zeros.
* Progress: scorer raw projected centerline progress in metres (clipped at zero),
  plus original normalized EP/PDM. This is route advance, not path length. Also
  save each intervention's lost metres and fraction of DS advance (null if DS
  advance is zero), since one metre can mean very different relative losses.
  Keep mean speed and stationary fraction; standstill by itself is not a failure.
* Duration/area integrals use endpoint trapezoids, so a constant event over the
  full horizon lasts 4 seconds, not 4.1. Threshold crossing between samples is
  approximated; this is sampled evaluation, not an analytic continuous solution.

Every method has paired candidate-minus-DS deltas and session-macro bootstrap
intervals (2000 draws), for the full cohort and actual interventions. Null values
are excluded **pairwise** with scene/session coverage, never imputed. A conditional
clearance comparison cannot establish all-cohort safety. Intervals are descriptive,
unadjusted for many axes and a repeatedly inspected training sample.

Progress tradeoff tables use fixed lost-progress budgets of 0, 0.5, 1, 2 metres.
Eligibility requires a real intervention, no worse original five safety components
or lane keeping, no increase in outside area maximum/integral or outside/contact duration.
Separate descriptive benefits: late road clearance +0.1 m, late actor clearance
+0.25 m with full actor interval coverage in both trajectories, or full-horizon
jerk RMS down >=20% **and** >=0.2 m/s³. These are declared descriptors, not learned
thresholds, an aggregate objective or a deployable gate. An improvement on one
axis may coexist with a regression on another; inspect the full delta table.
None of these privileged future/map measurements enter Drex or online decisions.

## Run from msu270

No update of the Astra checkout or GPU allocation is needed. The bridge stages its
own tools into a new directory in the existing Astra container. Its original
repository, maps, caches and portable package must still exist at the bound paths.

```bash
cd ~/drivesuprim
git pull --ff-only
export DREX_RUNTIME_ROOT=/scratch/pr0201/drex_runtime
export DREX_CONTINUOUS_SOURCE=replacement_gpu2_v1
export DREX_CONTINUOUS_NAME=rollout_audit_continuous_v1

bash scripts/drex/run_continuous_audit.sh check &&
bash scripts/drex/run_continuous_audit.sh prepare &&
bash scripts/drex/run_continuous_audit.sh connect &&
bash scripts/drex/run_continuous_audit.sh start
```

`check` and `prepare` use a CPU Slurm step and the existing Drex environment on
the setup node. Preparation validates the original commitments, including all
replacement results, and can take minutes. It freezes the full audit request.
`start` runs with nohup: it first audits a hash-selected four-session pilot,
downloads and verifies its artifacts, then automatically continues the full
cohort **only if the pilot passes**. It holds a single bridge lock. You can leave
the SSH session after `start`; no seven-hour GPU worker budget is involved.
Use `pilot` instead of `start` to stop after the small pilot if preferred.

```bash
bash scripts/drex/run_continuous_audit.sh status
```

The live bridge log prints completed / required candidates and an approximate
ETA after simulation begins. The retrieved snapshot may still be the pilot
while full replay is running. Final output is the checksum-verified directory
named by `runs/rollout_audit_continuous_v1/latest_result.json`:

* `report.json`: parity, coverage, paired deltas and progress tradeoffs;
* `per_scene.csv`, `per_candidate.csv`: detailed measurements and deltas;
* `candidates/<token>-<local>/`: committed states, geometry frames and summary;
* `identity.json`: frozen cache/config/environment/source provenance.

Repeating `start` with the SAME name verifies and reuses completed candidates.
A recorded candidate failure is preserved and blocks silent retries. A parity
failure requires diagnosis; never loosen tolerance or remove a failed scene to
obtain a favorable result. Changed scientific inputs/tools require a new name.
Original experiment files and decisions are preserved. The transport caps uploads
at 4 MiB and returned artifacts at 90 MiB; no full-cohort camera previews are made.

## Local verification

```bash
python -m unittest discover -s scripts/drex -p test_continuous_audit.py -v
```

The tests use analytic motion, polygon edge cases and a mocked CPU pipeline;
they require no Drex, CUDA, nuPlan datasets or network. Polygon-specific tests skip
if Shapely is absent in the Drex environment; they run in the local test environment.
Real dataset acceptance remains the Astra pilot's strict raw-score parity check.
