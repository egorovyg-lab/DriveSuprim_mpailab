"""Compact stdlib display of the six-scene top16 clearance audit."""
import argparse
import json
from pathlib import Path


def show(report):
    if report['status'] != 'complete': return json.dumps(report,indent=2)
    def number(v): return 'NA' if v is None else f'{v:.3f}'
    lines=[f'STATUS: complete; scenes={report["scenes"]}; candidates={report["candidates"]}',
        f'Metric parity={report["metric_parity_passed"]}/96; known-pair state parity='
        f'{report["known_pair_state_parity_passed"]}/12; scorer calls this invocation={report["scorer_calls_this_invocation"]}',
        'Ranks refer to DS top16, rank 1=DS. Buffers use privileged geometry; best PDM is an oracle diagnostic.']
    for scene in report['scenes_summary']:
        lines.append(f'\n{scene["case_id"]} token={scene["token"]} DS PDM={scene["baseline_pdm"]:.4f} '
                     f'progress={scene["baseline_progress_m"]:.3f} m')
        lines.append(f'Road covered: {scene["road_covered_candidates"]}/16; LK pass: {scene["lane_passing_candidates"]}/16; '
                     f'no sampled contact: {scene["contact_free_candidates"]}/16. Counts overlap.')
        for buffer,data in scene['buffers'].items():
            lines.append(f'buffer={buffer}m eligible={data["eligible_ranks"]} '
                f'progress>=DS={data["progress_not_less_than_ds_ranks"]} '
                f'bestPDMrank={data["best_pdm_rank"]} PDM={number(data["best_pdm"])} '
                f'gain_pp={number(data["best_gain_pp"])} progress_m={number(data["best_pdm_progress_m"])} '
                f'max_progress_m={number(data["maximum_progress_m"])}')
    lines.append(report['note'])
    return '\n'.join(lines)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('pointer',type=Path);args=p.parse_args()
    root=Path(json.loads(args.pointer.read_text())['result']);print('Artifacts:',root)
    print(show(json.loads((root/'report.json').read_text())))
