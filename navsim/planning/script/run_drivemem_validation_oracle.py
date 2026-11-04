"""Post-evaluation oracle on the new validation cohort, never a model selection step."""
import os
from pathlib import Path

from navsim.planning.script.run_drivemem_development_oracle import audit


def adapt(sample, state, choices):
    if state["token"] != sample["token"] or state["choices"] != choices:
        raise RuntimeError("Frozen validation decisions/token mismatch")
    if choices["baseline"] != int(sample["selected_local"]):
        raise RuntimeError("Frozen validation baseline mismatch")
    if any(str(i) not in state["outcomes"] for i in choices.values()):
        raise RuntimeError("Frozen validation action scoring incomplete")
    return dict(decisions=dict(choices), outcomes=dict(state["outcomes"]))


def main(cfg):
    from navsim.planning.script.run_drivemem_memory import (
        verify_identity, read_json as read, write_json as write, sha, sample, evaluators,
    )
    root = Path(cfg.drivemem.root)
    verify_identity(cfg, root)
    evaluation = root / "frozen_evaluation"
    metadata = read(evaluation / "identity.json")
    for key, path in dict(protocol_sha256=root / "validation_protocol.json", features_sha256=root / "export_index.json",
        cache_index_sha256=root / "cache_index.json", vocab_sha256=root / "vocab.npz", splits_sha256=root / "splits.json",
        runner_sha256=Path(__file__).with_name("run_drivemem_frozen_validation.py")).items():
        if metadata[key] != sha(path):
            raise RuntimeError(f"Validation input changed: {key}")
    report = read(evaluation / "report.json")
    if report["status"] != "complete":
        raise RuntimeError("Complete frozen validation first")
    records = read(root / "splits.json")["records"]
    if len(records) != report["scenes"] or any(r["role"] != "development" for r in records):
        raise RuntimeError("Oracle scope differs from frozen cohort")
    samples = [sample(root, r) for r in records]
    decisions = read(evaluation / "decisions.json")
    paths = {s["token"]: evaluation / "scenes" / (s["token"] + ".json") for s in samples}
    saved = {s["token"]: adapt(s, read(paths[s["token"]]), decisions[s["token"]]) for s in samples}
    provenance = dict(schema=1, cohort="post-frozen-validation-diagnostic", runner_sha256=sha(Path(__file__)),
        helper_sha256=sha(Path(__file__).with_name("run_drivemem_development_oracle.py")),
        inputs={name: sha(root / name) for name in ("identity.json", "split_identity.json", "vocab.npz", "export_index.json", "cache_index.json")},
        evaluation_identity=sha(evaluation / "identity.json"), report_sha256=sha(evaluation / "report.json"),
        decisions_sha256=sha(evaluation / "decisions.json"), query_outcomes={t: sha(p) for t, p in paths.items()})
    engine = []

    def evaluate(token, candidate):
        if not engine:
            engine.append(evaluators(cfg, root)[0])
        return engine[0](token, candidate)

    audit(root, samples, saved, evaluate, int(os.environ.get("DRIVEMEM_ORACLE_TOPK", "16")),
          provenance, read, write, dry_run=str(cfg.drivemem.stage).endswith("-plan"))


if __name__ == "__main__":
    import hydra
    hydra.main(config_path="config/pdm_scoring", config_name="default_drivemem_memory", version_base=None)(main)()
