# Run the CPU pool audit independently on Astra

Scientific inputs, tool hashes, reuse, checkpointing and metrics are unchanged.
The new controller lives inside the Astra container and writes `direct.log` and
`direct_job.json`. It runs the pilot and then full replay, without an ongoing
SSH session or msu270 process. Closing either terminal does not stop it.
Container/host shutdown still interrupts work; restarting reuses scene checkpoints.

The frozen original pool and verified source metadata are on msu270, so prepare
and transfer them once there:

```bash
cd ~/drivesuprim
git pull --ff-only
bash scripts/drex/run_pool_direct.sh prepare &&
bash scripts/drex/run_pool_direct.sh connect &&
bash scripts/drex/run_pool_direct.sh stage
```

Then on the Astra HOST (not inside the container):

```bash
A=/workspace/exp_v2/drex_rollout_audit/rollout_audit_pool64_v1
docker exec -d egorov_drivesuprim python -u "$A/direct_worker.py" "$A" /workspace/DriveSuprim
docker exec egorov_drivesuprim tail -n 25 "$A/direct.log"
docker exec egorov_drivesuprim cat "$A/direct_job.json"
```

Alternatively, after preparation/authentication, `run_pool_direct.sh start` on
msu270 stages and launches the detached Astra controller, then returns immediately.
It refuses duplicate work when either the direct controller or an existing
scientific worker holds its lock. It does not stop existing workers.

After completion, on msu270:

```bash
bash "$HOME/drivesuprim/scripts/drex/run_pool_direct.sh" connect
bash "$HOME/drivesuprim/scripts/drex/run_pool_direct.sh" fetch
```

`fetch` requires a stopped, complete run, verifies request and archive checksums,
and creates the same `latest_result.json` used by `download_audit.ps1` on Windows.
`run_pool_direct.sh status` reads the live Astra state and recent log; unlike the
previous `run_pool_audit.sh status`, it does not require a retrieved result first.
No GPU, new inference, model setup, dataset transfer or Astra git update is needed.
