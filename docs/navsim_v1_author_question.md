# Draft: exact checkpoint and scoring protocol for NAVSIM v1 Table 2

This draft has not been posted or sent.

Hello, we are trying to reproduce Table 2 of arXiv:2506.06659v3.
With the released R34 checkpoint, we obtain 83.0677 EPDMS (83.1 rounded).
We have not reproduced the Table 2 result of 89.9 PDMS.

Could you clarify whether Table 2 uses the same released checkpoint, or
separately trained weights / a different checkpoint selection?

Released checkpoint SHA256:
73db58e4ed4353f2238623f11c9ab409cf02470929a585a1b425266077d52a21
(epoch 8, global_step 11970). All 872 state keys load without missing or
unexpected keys. Evaluation covers all 12146 official navtest tokens.

Our compatibility adaptation uses the original NAVSIM v1.1 scorer/cache
processor at 3e8291bfa89ff247231e0227778840cd0a036896, Table 8 inference
weights, teacher, one refinement stage, three layers, and top-K 256.
It is an adaptation, not a claim to have reproduced your exact v1 pipeline.

The released checkpoint has a history_comfort prediction head, with no
separate v1 comfort head. We use that prediction for candidate selection,
then compute actual comfort with the original v1 evaluator.

Results:
- Original v1.1 definitions: NC 97.3695, DAC 96.2868, EP 83.0635,
  TTC 90.8859, C 99.9918, PDMS 86.7300.
- Same predictions, simulated states, 5s observation cache and all other
  metrics, changing ONLY the TTC ego loop endpoint from 4.0s to 3.1s:
  TTC 96.5009, PDMS 88.9221. Baseline replay matches the saved per-scene CSV.
- Applying aggregate_old_pdms from grid_search_unlog_expand.py row-wise
  to our saved v2 CSV gives 88.8797 (using v2 EP/TTC/history_comfort).
  This is a different comparison, not a controlled single-factor test.

The apparent mean EP difference is largely a definition effect:
750 scenes have NC*DAC=0 in the v1 run; their EP is 0 in v1 but averages
85.99% in v2. In the 11384 scenes with v1 NC*DAC=1, mean EP is 88.5472%
in v1 and 88.4438% in v2. These CSVs may have different selected trajectories.

Additional controls:
- HumanAgent on the full v1 cache gives 94.5455 PDMS, versus 94.8 in Table 2.
- Two prediction sets differ on 48.18% of scenes, but score 86.7300 and
  86.7467 on the same v1 evaluator/cache.
- Teacher vocabulary differs from the configured vocabulary by at most
  0.71 mm; checkpoint and vocabulary were not modified.

Could you provide:
1. The exact checkpoint (ideally its hash) and command that produced Table 2.
2. The NAVSIM commit, scorer configuration and cache-generation setup.
3. Whether TTC uses ego positions through 4.0s or 3.1s, and whether EP and
   comfort are original v1 definitions or v2 submetrics with old aggregation.
4. Whether metric heads were trained with v1 or v2 labels for Table 2,
   and how the v1 C prediction is obtained.
5. Whether the reported HumanAgent 94.8 uses the same public setup.

The shorter-TTC result exceeds the paper's TTC (96.5 vs 93.6), so we do
not interpret its closer PDMS as proof of the paper protocol.
We aim to reproduce the experiment rather than tune settings to 89.9.
