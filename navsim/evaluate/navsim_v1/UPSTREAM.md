# NAVSIM v1 source provenance

Source: https://github.com/autonomousvision/navsim/tree/3e8291bfa89ff247231e0227778840cd0a036896
Branch used: v1.1. Frozen commit: 3e8291bfa89ff247231e0227778840cd0a036896.

Copied source files:
- pdm_scorer.py: upstream navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py
- pdm_comfort_metrics.py: upstream navsim/planning/simulation/planner/pdm_planner/scoring/pdm_comfort_metrics.py
- pdm_enums.py: upstream navsim/planning/simulation/planner/pdm_planner/utils/pdm_enums.py
- metric_cache_processor.py: upstream navsim/planning/metric_caching/metric_cache_processor.py

Adaptations:
- Enum and comfort imports use the private navsim.evaluate.navsim_v1 namespace.
- Cache processor uses a private cache dataclass containing the original v1 fields.
- The local PDMClosedPlanner reference scorer is replaced with the same v1 scorer,
  wrapped to return the DataFrames expected by the local planner.
- Shared local dataset, scenario, simulator, observation, geometry, and collision utilities
  remain integration dependencies. This is a v1 compatibility implementation inside
  DriveSuprim, not a claim that the entire original NAVSIM environment is identical.

Why v1.1: it has the NC/DAC product and EP/TTC/C weighted mean described in the
DriveSuprim paper, Appendix A. NAVSIM v1.0 also multiplied DDC and used a different
progress threshold. In v1.1 DDC has zero weight by default.

Important: original TTC needs observation through 4.9 s for a 4 s prediction.
The DriveSuprim v2 cache builder produces only 4 s of interpolated observations.
The separate v1 cache therefore uses 5 s observations and a v1 PDM-Closed reference.
Do not point NAVSIM_V1_CACHE at the v2 cache.
