# Full top64 continuous reserve audit

This is a CPU evaluation of all 651 x 64 = 41664 original TRAIN candidates,
not another Drex inference experiment or a new online policy. It uses the same
continuous definitions and exact nine-value raw parity gate as the completed
1041-candidate audit. The 1041 results are reused only after SHA, metric code,
environment, scorer, vehicle, cache, package and candidate/label checks. Up to
40623 additional candidates require simulation. Original experiments stay intact.

## Fixed questions

Compare the *available reserve* in nested original top16, top32 and top64 pools.
For each, assess four simultaneous absolute/relative progress-loss limits:
0 m / 0%, 0.5 m / 5%, 1 m / 5%, 2 m / 10%. One metre is not automatically a
small relative loss. Progress is the same clipped raw centerline advance as in
the original audit, not path length or normalized EP.

Two objectives are evaluated separately: minimum full-horizon physical jerk RMS
and minimum lateral-acceleration RMS. DS is always available and wins ties.
The primary reserve is top64, <=1 m **and** <=5% lost advance, jerk objective,
with `geometry_peaks` guards. Meaningful jerk improvement requires >=20% and
>=0.2 m/s³; meaningful lateral improvement requires >=20% and >=0.05 m/s².

Two explicitly different guard levels are included:

* `raw_only` is a permissive reference: no worse measured NC, DAC, DDC, TLC,
  TTC, lane keeping or history comfort, plus the progress limit. It deliberately
  does not establish geometric safety or non-regression of other comfort axes.
* `geometry_peaks` additionally forbids more offroad area maximum/integral,
  offroad time, geometric contact time, acceleration peaks, jerk peak, yaw-rate
  or yaw-acceleration peak. Longitudinal RMS and the other comfort objective's
  RMS must not worsen. Minimum road and actor clearance must not decrease in
  either the full or >=1 s windows. An offroad DS can only switch to a fully
  covered footprint. Actor interval coverage must be complete in both paths;
  missing comparisons reject the challenger. DS remains the fallback even if
  it is unsafe: relative guard satisfaction is not absolute safety.

These oracles use **measured futures**, not forecasts. Their results cannot be
reported as Drex or trained-selector performance. They quantify room for a
future predictor and provide potential TRAIN labels. No holdout is opened.
Report all predeclared conditions; do not select the best one as independent
validation. Session-macro bootstrap intervals remain descriptive after repeated
inspection of this archive-selected sample. Distances/contacts are sampled,
non-reactive, four-second geometry, not continuous collision guarantees.

## Launch on msu270

```bash
cd ~/drivesuprim
git pull --ff-only
export DREX_RUNTIME_ROOT=/scratch/pr0201/drex_runtime
export DREX_POOL_NAME=rollout_audit_pool64_v1

bash "$HOME/drivesuprim/scripts/drex/run_pool_audit.sh" check &&
bash "$HOME/drivesuprim/scripts/drex/run_pool_audit.sh" prepare &&
bash "$HOME/drivesuprim/scripts/drex/run_pool_audit.sh" connect &&
bash "$HOME/drivesuprim/scripts/drex/run_pool_audit.sh" start
```

Preparation uses CPU Slurm on msu270. It validates the complete original table
run and the verified previous continuous result archive. The original top64
label source remains authoritative, including candidates absent from top16 NPZ.
Tools are staged into a separate Astra audit directory; no Astra git update or
GPU allocation is required. `start` uses nohup and a lock, first runs all 64
candidates in each of four fixed pilot scenes, verifies the pilot export, then
automatically continues all scenes. `pilot` stops after the mini-run instead.
You may leave SSH after `start`. The full CPU replay has no artificial seven-hour
worker budget. A rough several-hour scale follows from the previous audit;
actual ETA is printed after scene checkpoints, and depends on CPU load.

```bash
bash "$HOME/drivesuprim/scripts/drex/run_pool_audit.sh" status
```

Repeat `start` with the same name to resume. Completed scene blocks are hash
verified and skipped. An interruption can redo at most the incomplete scene's
64 candidates. A recorded candidate/parity failure blocks silent retries; inspect
it before changing the protocol. Full estimates require every candidate.

## Outputs and scale

The verified result directory is in `latest_result.json`. `report.json` includes
all 48 predeclared reserve variants and a compact top16/32/64 primary-budget
overview. `per_scene.csv.gz` contains oracle choices and paired deltas;
`per_candidate.csv.gz` contains all measured raw and continuous metrics.
`oracle_choices.json` binds candidate IDs. Python/pandas can read `.csv.gz`
directly; do not paste the full tables into chat.

After updating the local Windows checkout, download from local PowerShell:

```powershell
& "D:\DriveSuprim_mpailab\scripts\drex\download_audit.ps1"
```

This verifies completed status and downloads the three result files into
`D:\DriveSuprim_mpailab\results\pool64_v1`. Pass `-Destination` to choose another
directory. Then provide that local path for code-based analysis; gzip files do
not need manual extraction.

Full states and scene commitments remain on Astra under the new audit's
`artifacts/scenes/` directory. Only compact reports and compressed CSV files
return to msu270. Compressed uploads are capped at 32 MiB (128 MiB expanded),
exports at 90 MiB; the old 4 MiB small-audit transport is unchanged.

Local tests use synthetic motion/pools and mocked scoring/SSH/Slurm. They verify
guard boundaries, missing-data fallback, nested pools, reuse, checkpoint tamper
detection, failure preservation, archive traversal checks and pilot-before-full
routing. Real acceptance still requires the Astra pilot's parity results.
