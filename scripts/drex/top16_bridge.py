"""Login-node stdlib bridge for all top16 candidates in the original six scenes."""
import argparse
from pathlib import Path

import clearance_bridge as bridge
from transfer import read, sha

TOOLS = ('top16_remote.py', *bridge.TOOLS)


def request_for(root, source, package):
    request = bridge.source_snapshot(root, source)
    original = read(root/'runs'/source/'audit_request.json')
    manifest = root/'data/navsim_train_v1/package/package_manifest.json'
    if sha(manifest) != original['package_manifest_sha256']:
        raise RuntimeError('msu270 package differs from the original six-scene audit')
    return dict(request, protocol='clearance_top16_v1', package=package,
                package_manifest_sha256=original['package_manifest_sha256'],
                note='All 16 DS-ranked candidates in each of the six original diagnostic scenes. '
                     'Selection frozen before reading top16 outcome arrays. CPU scoring/geometry only; no Drex or holdout.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--name', required=True)
    p.add_argument('--source', default='rollout_audit_pair6_v2')
    p.add_argument('--host', default='egorov@93.180.62.99'); p.add_argument('--port', type=int, default=27434)
    p.add_argument('--container', default='egorov_drivesuprim')
    p.add_argument('--control', type=Path, default=Path.home()/'.ssh/drex-astra-transfer.sock')
    p.add_argument('--remote-repo', default='/workspace/DriveSuprim')
    p.add_argument('--remote-package', default='/workspace/exp_v2/drex_transfer/navsim_train_v1')
    args = p.parse_args(); root = args.root.expanduser().resolve()
    request = request_for(root, args.source, args.remote_package)
    raise SystemExit(bridge.run(args, request=request, tools=TOOLS, remote_tool='top16_remote.py'))
