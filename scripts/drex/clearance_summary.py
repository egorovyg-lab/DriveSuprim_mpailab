"""Compact standard-library-only display of the clearance report on msu270."""
import argparse
import json
from pathlib import Path


def show(report):
    if report['status'] != 'complete':
        return json.dumps(report, indent=2)
    def number(value): return 'NA' if value is None else f'{value:.3f}'
    lines = [f'STATUS: complete; candidates={report["candidates"]}; traces={report["geometry_traces"]}',
        f'Original parity={report["original_metric_parity_passed"]}/12; '
        f'corner checks={report["corner_mask_checks_passed"]}/24; lane checks={report["lane_counter_checks_passed"]}/24',
        'case member path       cornerDAC fullBody roadMin(m) cornerMin(m) outside(m2) LK counter actorMin(m) contact(s)']
    for row in report['per_candidate']:
        lines.append(f'{row["pair_id"]} {row["member"]} {row["geometry"]:9s} '
            f'{int(row["pdm_corner_pass"]):9d} {int(row["full_footprint_covered_all_samples"]):8d} '
            f'{number(row["min_road_clearance_m"]):>10s} {number(row["min_corner_signed_margin_m"]):>12s} '
            f'{number(row["max_outside_area_m2"]):>11s} {int(row["lane_keeping"]):2d} '
            f'{row["max_lane_counter_samples"]:2d}/{row["lane_required_samples"]:<2d} '
            f'{number(row["min_actor_clearance_m"]):>11s} {number(row["first_actor_contact_time_s"]):>10s}')
    lines.append('ELIGIBLE MEMBERS: 0=DS, 1=alternative; []=neither; [0,1]=both. Privileged sampled-geometry diagnostic only.')
    for pair in report['pairs']:
        lines.append(pair['pair_id']+' '+json.dumps(pair['eligible_members_by_buffer_m'], sort_keys=True))
    lines += ['roadMin=NA when the full body is outside at any sample. cornerMin is a separate signed corner distance.',
              'actorMin=NA means no recorded objects; contact is geometric intersection, not fault/TTC.',
              'Includes t=0, step=0.1 s. Buffers 0/0.1/0.25/0.5 m are descriptive, not calibrated safety.',
              report['note']]
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('pointer', type=Path)
    args = parser.parse_args(); root = Path(json.loads(args.pointer.read_text())['result'])
    print('Artifacts:', root)
    print(show(json.loads((root/'report.json').read_text())))
