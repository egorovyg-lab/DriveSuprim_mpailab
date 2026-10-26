# Draft: reproducing NAVSIM v1 Table 2 with the released R34 checkpoint

This draft has not been posted or sent.

Hello, thank you for releasing DriveSuprim.

We obtain 83.0677 EPDMS with the released R34 checkpoint, matching 83.1
after rounding, but have not reproduced the NAVSIM v1 PDMS of 89.9 in
Table 2 of arXiv:2506.06659v3.

Is the released drivesuprim_r34.ckpt also the checkpoint used for Table 2,
or does the v1 experiment use separately trained weights or a different
checkpoint selection?

Checkpoint SHA256:
73db58e4ed4353f2238623f11c9ab409cf02470929a585a1b425266077d52a21
(epoch 8, global_step 11970).

We made an explicit compatibility adaptation, rather than just disabling
v2 metric weights:
- NAVSIM v1.1 scorer/cache processor pinned to
  3e8291bfa89ff247231e0227778840cd0a036896;
- 5-second observations for full-horizon TTC;
- NC * DAC * (5 EP + 5 TTC + 2 C) / 12;
- Table 8 inference coefficients on both coarse and refinement stages:
  imitation 0.05, NC 0.5, DAC 0.5, EP 5, TTC 5, C 2, lambda_avg 8;
- teacher, one refinement stage, three layers, top-K 256.

The released model has a history_comfort head and no separate C head.
Our adaptation currently uses the history_comfort prediction for C when
ranking candidates, while evaluating actual comfort with the v1 function.
We do not know whether this matches your Table 2 procedure.

On all 12146 navtest scenes with zero failures we obtain:
NC 97.3695, DAC 96.2868, EP 83.0635, TTC 90.8859, C 99.9918,
PDMS 86.7300.

Checks completed:
- all 872 checkpoint state keys load, with no missing/unexpected keys;
- the local split equals the official v1.1/released navtest token set;
- HumanAgent gives 94.5455 PDMS, compared with 94.8 in Table 2;
- two saved prediction sets differ on 48.18% of scenes but give 86.7300
  and 86.7467 on the same v1 scorer;
- substituting reference trajectories/centerlines from a pre-existing
  cache has no visible score effect on a deterministic 256-scene sample;
- teacher vocabulary differs by at most 0.71 mm; nearest-index identity
  matches for 64/64 sampled trajectories. The checkpoint was not changed.

Could you confirm or provide:
1. The exact checkpoint and command for the v1 result in Table 2.
2. The NAVSIM commit, scorer config and cache-generation setup.
3. How the C prediction in Table 8 is obtained from the released model.
4. Whether the HumanAgent 94.8 uses the same public navtest/cache setup.

We aim to reproduce the original experiment, not tune evaluation settings
to approach the reported score.
