"""Build/audit the 399-scene training union. No training, GPU, or holdout labels."""
import argparse
import json
import os
from pathlib import Path

from navsim.agents import selector_training_bundle as bundle
from navsim.planning.script.run_drivemem_expanded_utility import write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    base = Path(os.environ.get("NAVSIM_EXP_ROOT", "/workspace/exp_v2")) / "drivemem_memory"
    p.add_argument("stage", choices=("build", "audit"))
    p.add_argument("--prior", type=Path, default=base / "experiments/pair_utility_v1")
    p.add_argument("--addition", type=Path, default=base / "mf_new4_shard174_v1")
    p.add_argument("--decoder", type=Path, default=base / "decoder_features_functional_v1")
    p.add_argument("--collection", type=Path, default=base / "selector_collection_v2")
    p.add_argument("--out", type=Path, default=base / "selector_training_bundle_v2")
    args = p.parse_args()
    prior, addition, decoder, collection, out = [v.resolve() for v in (
        args.prior, args.addition, args.decoder, args.collection, args.out)]
    for root in (prior, addition, decoder, collection):
        if out == root or out in root.parents or root in out.parents:
            raise ValueError("Separate training-bundle output root required")
    if args.stage == "build":
        examples, vocab, inputs, protected = bundle.load_sources(prior, addition, decoder, collection)
        bundle.build(out, examples, vocab, inputs, protected)
    examples, _ = bundle.load_bundle(out)
    audit, rows = bundle.audit(examples)
    write(out / "audit.json", audit)
    # JSON keeps exact scene IDs/types and all source origins for diagnostics.
    write(out / "audit_per_scene.json", rows)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
