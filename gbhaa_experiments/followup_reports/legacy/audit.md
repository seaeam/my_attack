# Legacy evidence audit

Read-only inventory; no experiments were executed.

## Counts

```json
{
  "records": 177,
  "expected_cells": 210,
  "missing_cells": 37,
  "unusable_cells": 16,
  "usable_records": 157,
  "complete_seed_blocks": 47,
  "usable_complete_seed_blocks": 44,
  "factorial_settings_mismatches": 0,
  "unresolved_mapping": 19,
  "issue_counts": {
    "cell_mapping_conflict": 12,
    "drop_disagreement": 5,
    "duplicate_cell": 8,
    "historical_provenance_unknown": 177,
    "misclassification_disagreement": 5,
    "path_mapping_unresolved": 4,
    "uniform_global_invariance": 19
  }
}
```

## Scope

```json
{
  "ae_ppr": {
    "datasets": [
      "citeseer"
    ],
    "budgets": [
      0.05,
      0.1
    ],
    "seeds": [
      15,
      16,
      17,
      18,
      19
    ],
    "variants": {
      "citeseer": [
        "ppr_ppr",
        "ae_ppr_ppr",
        "ppr_ae_ppr",
        "ae_ppr_ae_ppr"
      ]
    }
  },
  "hybrid": {
    "datasets": [
      "citeseer"
    ],
    "budgets": [
      0.05,
      0.1
    ],
    "seeds": [
      15,
      16,
      17,
      18,
      19
    ],
    "variants": {
      "citeseer": [
        "edge_only",
        "feature_only",
        "parallel",
        "serial"
      ]
    }
  },
  "gb_ablation": {
    "datasets": [
      "citeseer"
    ],
    "budgets": [
      0.05,
      0.1
    ],
    "seeds": [
      15,
      16,
      17,
      18,
      19
    ],
    "variants": {
      "citeseer": [
        "gb",
        "kmeans",
        "random",
        "node_level"
      ]
    }
  },
  "efficiency": {
    "datasets": [
      "citeseer",
      "cora"
    ],
    "budgets": [
      0.01,
      0.05,
      0.1
    ],
    "seeds": [
      15,
      16,
      17,
      18,
      19
    ],
    "variants": {
      "citeseer": [
        "gb",
        "kmeans",
        "node_level"
      ],
      "cora": [
        "gb",
        "kmeans",
        "node_level"
      ]
    }
  }
}
```

## Evidence limits and plan concerns

- This is a descriptive inventory of recorded summaries, not verification of manuscript tables/figures or paper provenance; no automatic number replacement is justified.
- The current default is fixed to edge-first in heir.py; the explicitly named pre-flip diagnostic retains the former attribute-before-edge behavior. Historical JSON has no verifiable source snapshot, so historical execution-order provenance remains unknown. Version edge-first follow-up results separately and never pool them with legacy runs.
- The reviewed global operator accumulates iterates using row-normalized P @ z; it is not established as standard stationary PageRank. A uniform source is invariant under this operator (up to numerical effects); degree-source runs are not thereby invalid. Transition labels alone do not establish the historical operator.
- Only explicit recorded command options are extracted; omitted options are unknown, not inferred from current defaults. Matching recorded settings cannot prove matching unrecorded defaults, code, graph/split/features, completed budgets, or RNG state.
- Usable complete seed blocks mean all expected arms have unique, internally consistent recorded summaries. AE-PPR blocks also require matched recorded settings except the two transition flags. This does not establish causal effects or publication readiness.
- The four-arm factorial may have different structural trajectories; strict selector controls instead require the same candidate universe, frozen structure features, matched trajectory hashes, and aligned victim RNG/predictions. Legacy records cannot verify these requirements. Shared numeric seeds alone do not prove pairing.
- Efficiency comparisons require matched hardware/runtime, model/cache conditions, and completed work. Runtime metadata is retained, but recorded elapsed time alone cannot establish a speedup. Configuration groups separate recorded profiles, without pooling.
- Cell identity uses directory coordinates when parseable; conflicting metadata remains unresolved. Irregular paths retain metadata claims but are unusable. Missing cells have no file mapped under this rule, distinct from present but unusable cells; a misplaced record cannot be reassigned to a missing cell without resolving its provenance. runs.jsonl and stdout logs are deliberately excluded. Metric consistency tolerance is an absolute 0.0003 for rounded accuracy summaries.

## Exact missing cells

- `ae_ppr/citeseer/0.1/19/ppr_ppr`
- `ae_ppr/citeseer/0.1/19/ae_ppr_ppr`
- `ae_ppr/citeseer/0.1/19/ppr_ae_ppr`
- `ae_ppr/citeseer/0.1/19/ae_ppr_ae_ppr`
- `hybrid/citeseer/0.1/18/feature_only`
- `hybrid/citeseer/0.1/18/parallel`
- `hybrid/citeseer/0.1/18/serial`
- `hybrid/citeseer/0.1/19/edge_only`
- `hybrid/citeseer/0.1/19/feature_only`
- `hybrid/citeseer/0.1/19/parallel`
- `hybrid/citeseer/0.1/19/serial`
- `efficiency/cora/0.05/16/kmeans`
- `efficiency/cora/0.05/16/node_level`
- `efficiency/cora/0.05/17/gb`
- `efficiency/cora/0.05/17/kmeans`
- `efficiency/cora/0.05/17/node_level`
- `efficiency/cora/0.05/18/gb`
- `efficiency/cora/0.05/18/kmeans`
- `efficiency/cora/0.05/18/node_level`
- `efficiency/cora/0.05/19/gb`
- `efficiency/cora/0.05/19/kmeans`
- `efficiency/cora/0.05/19/node_level`
- `efficiency/cora/0.1/15/gb`
- `efficiency/cora/0.1/15/kmeans`
- `efficiency/cora/0.1/15/node_level`
- `efficiency/cora/0.1/16/gb`
- `efficiency/cora/0.1/16/kmeans`
- `efficiency/cora/0.1/16/node_level`
- `efficiency/cora/0.1/17/gb`
- `efficiency/cora/0.1/17/kmeans`
- `efficiency/cora/0.1/17/node_level`
- `efficiency/cora/0.1/18/gb`
- `efficiency/cora/0.1/18/kmeans`
- `efficiency/cora/0.1/18/node_level`
- `efficiency/cora/0.1/19/gb`
- `efficiency/cora/0.1/19/kmeans`
- `efficiency/cora/0.1/19/node_level`

## Present but unusable cells

- `ae_ppr/citeseer/0.1/15/ppr_ppr` — duplicate_cell, path_mapping_unresolved
- `ae_ppr/citeseer/0.1/15/ae_ppr_ppr` — duplicate_cell, path_mapping_unresolved
- `ae_ppr/citeseer/0.1/15/ppr_ae_ppr` — duplicate_cell, path_mapping_unresolved
- `ae_ppr/citeseer/0.1/15/ae_ppr_ae_ppr` — drop_disagreement, duplicate_cell, misclassification_disagreement, path_mapping_unresolved
- `ae_ppr/citeseer/0.1/16/ppr_ppr` — cell_mapping_conflict
- `ae_ppr/citeseer/0.1/16/ae_ppr_ppr` — cell_mapping_conflict
- `ae_ppr/citeseer/0.1/16/ppr_ae_ppr` — cell_mapping_conflict
- `ae_ppr/citeseer/0.1/16/ae_ppr_ae_ppr` — cell_mapping_conflict, drop_disagreement, misclassification_disagreement
- `ae_ppr/citeseer/0.1/17/ppr_ppr` — cell_mapping_conflict
- `ae_ppr/citeseer/0.1/17/ae_ppr_ppr` — cell_mapping_conflict
- `ae_ppr/citeseer/0.1/17/ppr_ae_ppr` — cell_mapping_conflict
- `ae_ppr/citeseer/0.1/17/ae_ppr_ae_ppr` — cell_mapping_conflict, drop_disagreement, misclassification_disagreement
- `ae_ppr/citeseer/0.1/18/ppr_ppr` — cell_mapping_conflict
- `ae_ppr/citeseer/0.1/18/ae_ppr_ppr` — cell_mapping_conflict
- `ae_ppr/citeseer/0.1/18/ppr_ae_ppr` — cell_mapping_conflict
- `ae_ppr/citeseer/0.1/18/ae_ppr_ae_ppr` — cell_mapping_conflict, drop_disagreement, misclassification_disagreement

## Usable complete seed blocks (descriptive only)

- ae_ppr/citeseer/0.05/seed_15: ppr_ppr, ae_ppr_ppr, ppr_ae_ppr, ae_ppr_ae_ppr; configuration group `73ae8922cdf6cbd3`
- ae_ppr/citeseer/0.05/seed_16: ppr_ppr, ae_ppr_ppr, ppr_ae_ppr, ae_ppr_ae_ppr; configuration group `73ae8922cdf6cbd3`
- ae_ppr/citeseer/0.05/seed_17: ppr_ppr, ae_ppr_ppr, ppr_ae_ppr, ae_ppr_ae_ppr; configuration group `73ae8922cdf6cbd3`
- ae_ppr/citeseer/0.05/seed_18: ppr_ppr, ae_ppr_ppr, ppr_ae_ppr, ae_ppr_ae_ppr; configuration group `73ae8922cdf6cbd3`
- ae_ppr/citeseer/0.05/seed_19: ppr_ppr, ae_ppr_ppr, ppr_ae_ppr, ae_ppr_ae_ppr; configuration group `73ae8922cdf6cbd3`
- hybrid/citeseer/0.05/seed_15: edge_only, feature_only, parallel, serial; configuration group `13fe1c73abab8cfb`
- hybrid/citeseer/0.05/seed_16: edge_only, feature_only, parallel, serial; configuration group `13fe1c73abab8cfb`
- hybrid/citeseer/0.05/seed_17: edge_only, feature_only, parallel, serial; configuration group `13fe1c73abab8cfb`
- hybrid/citeseer/0.05/seed_18: edge_only, feature_only, parallel, serial; configuration group `13fe1c73abab8cfb`
- hybrid/citeseer/0.05/seed_19: edge_only, feature_only, parallel, serial; configuration group `13fe1c73abab8cfb`
- hybrid/citeseer/0.1/seed_15: edge_only, feature_only, parallel, serial; configuration group `13fe1c73abab8cfb`
- hybrid/citeseer/0.1/seed_16: edge_only, feature_only, parallel, serial; configuration group `13fe1c73abab8cfb`
- hybrid/citeseer/0.1/seed_17: edge_only, feature_only, parallel, serial; configuration group `13fe1c73abab8cfb`
- gb_ablation/citeseer/0.05/seed_15: gb, kmeans, random, node_level; configuration group `4057323b56db6482`
- gb_ablation/citeseer/0.05/seed_16: gb, kmeans, random, node_level; configuration group `4057323b56db6482`
- gb_ablation/citeseer/0.05/seed_17: gb, kmeans, random, node_level; configuration group `4057323b56db6482`
- gb_ablation/citeseer/0.05/seed_18: gb, kmeans, random, node_level; configuration group `4057323b56db6482`
- gb_ablation/citeseer/0.05/seed_19: gb, kmeans, random, node_level; configuration group `4057323b56db6482`
- gb_ablation/citeseer/0.1/seed_15: gb, kmeans, random, node_level; configuration group `4057323b56db6482`
- gb_ablation/citeseer/0.1/seed_16: gb, kmeans, random, node_level; configuration group `4057323b56db6482`
- gb_ablation/citeseer/0.1/seed_17: gb, kmeans, random, node_level; configuration group `4057323b56db6482`
- gb_ablation/citeseer/0.1/seed_18: gb, kmeans, random, node_level; configuration group `4057323b56db6482`
- gb_ablation/citeseer/0.1/seed_19: gb, kmeans, random, node_level; configuration group `4057323b56db6482`
- efficiency/citeseer/0.01/seed_15: gb, kmeans, node_level; configuration group `3420d5b58577da15`
- efficiency/citeseer/0.01/seed_16: gb, kmeans, node_level; configuration group `3420d5b58577da15`
- efficiency/citeseer/0.01/seed_17: gb, kmeans, node_level; configuration group `3420d5b58577da15`
- efficiency/citeseer/0.01/seed_18: gb, kmeans, node_level; configuration group `3420d5b58577da15`
- efficiency/citeseer/0.01/seed_19: gb, kmeans, node_level; configuration group `3420d5b58577da15`
- efficiency/citeseer/0.05/seed_15: gb, kmeans, node_level; configuration group `3420d5b58577da15`
- efficiency/citeseer/0.05/seed_16: gb, kmeans, node_level; configuration group `3420d5b58577da15`
- efficiency/citeseer/0.05/seed_17: gb, kmeans, node_level; configuration group `3420d5b58577da15`
- efficiency/citeseer/0.05/seed_18: gb, kmeans, node_level; configuration group `3420d5b58577da15`
- efficiency/citeseer/0.05/seed_19: gb, kmeans, node_level; configuration group `3420d5b58577da15`
- efficiency/citeseer/0.1/seed_15: gb, kmeans, node_level; configuration group `3420d5b58577da15`
- efficiency/citeseer/0.1/seed_16: gb, kmeans, node_level; configuration group `3420d5b58577da15`
- efficiency/citeseer/0.1/seed_17: gb, kmeans, node_level; configuration group `3420d5b58577da15`
- efficiency/citeseer/0.1/seed_18: gb, kmeans, node_level; configuration group `3420d5b58577da15`
- efficiency/citeseer/0.1/seed_19: gb, kmeans, node_level; configuration group `3420d5b58577da15`
- efficiency/cora/0.01/seed_15: gb, kmeans, node_level; configuration group `f35d320fd8c622e6`
- efficiency/cora/0.01/seed_16: gb, kmeans, node_level; configuration group `f35d320fd8c622e6`
- efficiency/cora/0.01/seed_17: gb, kmeans, node_level; configuration group `f35d320fd8c622e6`
- efficiency/cora/0.01/seed_18: gb, kmeans, node_level; configuration group `f35d320fd8c622e6`
- efficiency/cora/0.01/seed_19: gb, kmeans, node_level; configuration group `f35d320fd8c622e6`
- efficiency/cora/0.05/seed_15: gb, kmeans, node_level; configuration group `f35d320fd8c622e6`

## Factorial settings mismatches

```json
[]
```

## Unexpected cells

```json
[]
```

## Unresolved mapping

```json
[
  {
    "path": "ae_ppr/citeseer/ptb_0.1/seed_16/ae_ppr_ae_ppr/result.json",
    "reason": "cell_mapping_conflict",
    "detail": {
      "path": {
        "experiment": "ae_ppr",
        "dataset": "citeseer",
        "ptb_rate": 0.1,
        "seed": 16,
        "variant": "ae_ppr_ae_ppr"
      },
      "metadata": {
        "experiment": "ae_ppr",
        "dataset": "citeseer",
        "ptb_rate": 0.1,
        "seed": 15,
        "variant": "ae_ppr_ae_ppr"
      }
    }
  },
  {
    "path": "ae_ppr/citeseer/ptb_0.1/seed_16/ae_ppr_ppr/result.json",
    "reason": "cell_mapping_conflict",
    "detail": {
      "path": {
        "experiment": "ae_ppr",
        "dataset": "citeseer",
        "ptb_rate": 0.1,
        "seed": 16,
        "variant": "ae_ppr_ppr"
      },
      "metadata": {
        "experiment": "ae_ppr",
        "dataset": "citeseer",
        "ptb_rate": 0.1,
        "seed": 15,
        "variant": "ae_ppr_ppr"
      }
    }
  },
  {
    "path": "ae_ppr/citeseer/ptb_0.1/seed_16/ppr_ae_ppr/result.json",
    "reason": "cell_mapping_conflict",
    "detail": {
      "path": {
        "experiment": "ae_ppr",
        "dataset": "citeseer",
        "ptb_rate": 0.1,
        "seed": 16,
        "variant": "ppr_ae_ppr"
      },
      "metadata": {
        "experiment": "ae_ppr",
        "dataset": "citeseer",
        "ptb_rate": 0.1,
        "seed": 15,
        "variant": "ppr_ae_ppr"
      }
    }
  },
  {
    "path": "ae_ppr/citeseer/ptb_0.1/seed_16/ppr_ppr/result.json",
    "reason": "cell_mapping_conflict",
    "detail": {
      "path": {
        "experiment": "ae_ppr",
        "dataset": "citeseer",
        "ptb_rate": 0.1,
        "seed": 16,
        "variant": "ppr_ppr"
      },
      "metadata": {
        "experiment": "ae_ppr",
        "dataset": "citeseer",
        "ptb_rate": 0.1,
        "seed": 15,
        "variant": "ppr_ppr"
      }
    }
  },
  {
    "path": "ae_ppr/citeseer/ptb_0.1/seed_17/ae_ppr_ae_ppr/result.json",
    "reason": "cell_mapping_conflict",
    "detail": {
      "path": {
        "experiment": "ae_ppr",
        "dataset": "citeseer",
        "ptb_rate": 0.1,
        "seed": 17,
        "variant": "ae_ppr_ae_ppr"
      },
      "metadata": {
        "experiment": "ae_ppr",
        "dataset": "citeseer",
        "ptb_rate": 0.1,
        "seed": 15,
        "variant": "ae_ppr_ae_ppr"
      }
    }
  },
  {
    "path": "ae_ppr/citeseer/ptb_0.1/seed_17/ae_ppr_ppr/result.json",
    "reason": "cell_mapping_conflict",
    "detail": {
      "path": {
        "experiment": "ae_ppr",
        "dataset": "citeseer",
        "ptb_rate": 0.1,
        "seed": 17,
        "variant": "ae_ppr_ppr"
      },
      "metadata": {
        "experiment": "ae_ppr",
        "dataset": "citeseer",
        "ptb_rate": 0.1,
        "seed": 15,
        "variant": "ae_ppr_ppr"
      }
    }
  },
  {
    "path": "ae_ppr/citeseer/ptb_0.1/seed_17/ppr_ae_ppr/result.json",
    "reason": "cell_mapping_conflict",
    "detail": {
      "path": {
        "experiment": "ae_ppr",
        "dataset": "citeseer",
        "ptb_rate": 0.1,
        "seed": 17,
        "variant": "ppr_ae_ppr"
      },
      "metadata": {
        "experiment": "ae_ppr",
        "dataset": "citeseer",
        "ptb_rate": 0.1,
        "seed": 15,
        "variant": "ppr_ae_ppr"
      }
    }
  },
  {
    "path": "ae_ppr/citeseer/ptb_0.1/seed_17/ppr_ppr/result.json",
    "reason": "cell_mapping_conflict",
    "detail": {
      "path": {
        "experiment": "ae_ppr",
        "dataset": "citeseer",
        "ptb_rate": 0.1,
        "seed": 17,
        "variant": "ppr_ppr"
      },
      "metadata": {
        "experiment": "ae_ppr",
        "dataset": "citeseer",
        "ptb_rate": 0.1,
        "seed": 15,
        "variant": "ppr_ppr"
      }
    }
  },
  {
    "path": "ae_ppr/citeseer/ptb_0.1/seed_17/seed_15/ae_ppr_ae_ppr/result.json",
    "reason": "path_mapping_unresolved",
    "detail": "Irregular directory layout; metadata claim retained, not verified"
  },
  {
    "path": "ae_ppr/citeseer/ptb_0.1/seed_17/seed_15/ae_ppr_ppr/result.json",
    "reason": "path_mapping_unresolved",
    "detail": "Irregular directory layout; metadata claim retained, not verified"
  },
  {
    "path": "ae_ppr/citeseer/ptb_0.1/seed_17/seed_15/ppr_ae_ppr/result.json",
    "reason": "path_mapping_unresolved",
    "detail": "Irregular directory layout; metadata claim retained, not verified"
  },
  {
    "path": "ae_ppr/citeseer/ptb_0.1/seed_17/seed_15/ppr_ppr/result.json",
    "reason": "path_mapping_unresolved",
    "detail": "Irregular directory layout; metadata claim retained, not verified"
  },
  {
    "path": "ae_ppr/citeseer/ptb_0.1/seed_18/ae_ppr_ae_ppr/result.json",
    "reason": "cell_mapping_conflict",
    "detail": {
      "path": {
        "experiment": "ae_ppr",
        "dataset": "citeseer",
        "ptb_rate": 0.1,
        "seed": 18,
        "variant": "ae_ppr_ae_ppr"
      },
      "metadata": {
        "experiment": "ae_ppr",
        "dataset": "citeseer",
        "ptb_rate": 0.1,
        "seed": 15,
        "variant": "ae_ppr_ae_ppr"
      }
    }
  },
  {
    "path": "ae_ppr/citeseer/ptb_0.1/seed_18/ae_ppr_ppr/result.json",
    "reason": "cell_mapping_conflict",
    "detail": {
      "path": {
        "experiment": "ae_ppr",
        "dataset": "citeseer",
        "ptb_rate": 0.1,
        "seed": 18,
        "variant": "ae_ppr_ppr"
      },
      "metadata": {
        "experiment": "ae_ppr",
        "dataset": "citeseer",
        "ptb_rate": 0.1,
        "seed": 15,
        "variant": "ae_ppr_ppr"
      }
    }
  },
  {
    "path": "ae_ppr/citeseer/ptb_0.1/seed_18/ppr_ae_ppr/result.json",
    "reason": "cell_mapping_conflict",
    "detail": {
      "path": {
        "experiment": "ae_ppr",
        "dataset": "citeseer",
        "ptb_rate": 0.1,
        "seed": 18,
        "variant": "ppr_ae_ppr"
      },
      "metadata": {
        "experiment": "ae_ppr",
        "dataset": "citeseer",
        "ptb_rate": 0.1,
        "seed": 15,
        "variant": "ppr_ae_ppr"
      }
    }
  },
  {
    "path": "ae_ppr/citeseer/ptb_0.1/seed_18/ppr_ppr/result.json",
    "reason": "cell_mapping_conflict",
    "detail": {
      "path": {
        "experiment": "ae_ppr",
        "dataset": "citeseer",
        "ptb_rate": 0.1,
        "seed": 18,
        "variant": "ppr_ppr"
      },
      "metadata": {
        "experiment": "ae_ppr",
        "dataset": "citeseer",
        "ptb_rate": 0.1,
        "seed": 15,
        "variant": "ppr_ppr"
      }
    }
  },
  {
    "reason": "historical_source_and_order",
    "detail": "No verified link from result JSON to an executed source snapshot/order",
    "paths": [
      "ae_ppr/citeseer/ptb_0.05/seed_15/ae_ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_15/ae_ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_15/ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_15/ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_16/ae_ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_16/ae_ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_16/ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_16/ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_17/ae_ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_17/ae_ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_17/ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_17/ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_18/ae_ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_18/ae_ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_18/ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_18/ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_19/ae_ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_19/ae_ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_19/ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_19/ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_15/ae_ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_15/ae_ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_15/ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_15/ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_16/ae_ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_16/ae_ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_16/ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_16/ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_17/ae_ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_17/ae_ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_17/ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_17/ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_17/seed_15/ae_ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_17/seed_15/ae_ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_17/seed_15/ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_17/seed_15/ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_18/ae_ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_18/ae_ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_18/ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_18/ppr_ppr/result.json",
      "hybrid/citeseer/ptb_0.05/seed_15/edge_only/result.json",
      "hybrid/citeseer/ptb_0.05/seed_15/feature_only/result.json",
      "hybrid/citeseer/ptb_0.05/seed_15/parallel/result.json",
      "hybrid/citeseer/ptb_0.05/seed_15/serial/result.json",
      "hybrid/citeseer/ptb_0.05/seed_16/edge_only/result.json",
      "hybrid/citeseer/ptb_0.05/seed_16/feature_only/result.json",
      "hybrid/citeseer/ptb_0.05/seed_16/parallel/result.json",
      "hybrid/citeseer/ptb_0.05/seed_16/serial/result.json",
      "hybrid/citeseer/ptb_0.05/seed_17/edge_only/result.json",
      "hybrid/citeseer/ptb_0.05/seed_17/feature_only/result.json",
      "hybrid/citeseer/ptb_0.05/seed_17/parallel/result.json",
      "hybrid/citeseer/ptb_0.05/seed_17/serial/result.json",
      "hybrid/citeseer/ptb_0.05/seed_18/edge_only/result.json",
      "hybrid/citeseer/ptb_0.05/seed_18/feature_only/result.json",
      "hybrid/citeseer/ptb_0.05/seed_18/parallel/result.json",
      "hybrid/citeseer/ptb_0.05/seed_18/serial/result.json",
      "hybrid/citeseer/ptb_0.05/seed_19/edge_only/result.json",
      "hybrid/citeseer/ptb_0.05/seed_19/feature_only/result.json",
      "hybrid/citeseer/ptb_0.05/seed_19/parallel/result.json",
      "hybrid/citeseer/ptb_0.05/seed_19/serial/result.json",
      "hybrid/citeseer/ptb_0.1/seed_15/edge_only/result.json",
      "hybrid/citeseer/ptb_0.1/seed_15/feature_only/result.json",
      "hybrid/citeseer/ptb_0.1/seed_15/parallel/result.json",
      "hybrid/citeseer/ptb_0.1/seed_15/serial/result.json",
      "hybrid/citeseer/ptb_0.1/seed_16/edge_only/result.json",
      "hybrid/citeseer/ptb_0.1/seed_16/feature_only/result.json",
      "hybrid/citeseer/ptb_0.1/seed_16/parallel/result.json",
      "hybrid/citeseer/ptb_0.1/seed_16/serial/result.json",
      "hybrid/citeseer/ptb_0.1/seed_17/edge_only/result.json",
      "hybrid/citeseer/ptb_0.1/seed_17/feature_only/result.json",
      "hybrid/citeseer/ptb_0.1/seed_17/parallel/result.json",
      "hybrid/citeseer/ptb_0.1/seed_17/serial/result.json",
      "hybrid/citeseer/ptb_0.1/seed_18/edge_only/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_15/gb/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_15/kmeans/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_15/node_level/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_15/random/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_16/gb/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_16/kmeans/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_16/node_level/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_16/random/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_17/gb/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_17/kmeans/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_17/node_level/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_17/random/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_18/gb/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_18/kmeans/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_18/node_level/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_18/random/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_19/gb/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_19/kmeans/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_19/node_level/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_19/random/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_15/gb/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_15/kmeans/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_15/node_level/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_15/random/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_16/gb/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_16/kmeans/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_16/node_level/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_16/random/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_17/gb/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_17/kmeans/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_17/node_level/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_17/random/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_18/gb/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_18/kmeans/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_18/node_level/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_18/random/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_19/gb/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_19/kmeans/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_19/node_level/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_19/random/result.json",
      "efficiency/citeseer/ptb_0.01/seed_15/gb/result.json",
      "efficiency/citeseer/ptb_0.01/seed_15/kmeans/result.json",
      "efficiency/citeseer/ptb_0.01/seed_15/node_level/result.json",
      "efficiency/citeseer/ptb_0.01/seed_16/gb/result.json",
      "efficiency/citeseer/ptb_0.01/seed_16/kmeans/result.json",
      "efficiency/citeseer/ptb_0.01/seed_16/node_level/result.json",
      "efficiency/citeseer/ptb_0.01/seed_17/gb/result.json",
      "efficiency/citeseer/ptb_0.01/seed_17/kmeans/result.json",
      "efficiency/citeseer/ptb_0.01/seed_17/node_level/result.json",
      "efficiency/citeseer/ptb_0.01/seed_18/gb/result.json",
      "efficiency/citeseer/ptb_0.01/seed_18/kmeans/result.json",
      "efficiency/citeseer/ptb_0.01/seed_18/node_level/result.json",
      "efficiency/citeseer/ptb_0.01/seed_19/gb/result.json",
      "efficiency/citeseer/ptb_0.01/seed_19/kmeans/result.json",
      "efficiency/citeseer/ptb_0.01/seed_19/node_level/result.json",
      "efficiency/citeseer/ptb_0.05/seed_15/gb/result.json",
      "efficiency/citeseer/ptb_0.05/seed_15/kmeans/result.json",
      "efficiency/citeseer/ptb_0.05/seed_15/node_level/result.json",
      "efficiency/citeseer/ptb_0.05/seed_16/gb/result.json",
      "efficiency/citeseer/ptb_0.05/seed_16/kmeans/result.json",
      "efficiency/citeseer/ptb_0.05/seed_16/node_level/result.json",
      "efficiency/citeseer/ptb_0.05/seed_17/gb/result.json",
      "efficiency/citeseer/ptb_0.05/seed_17/kmeans/result.json",
      "efficiency/citeseer/ptb_0.05/seed_17/node_level/result.json",
      "efficiency/citeseer/ptb_0.05/seed_18/gb/result.json",
      "efficiency/citeseer/ptb_0.05/seed_18/kmeans/result.json",
      "efficiency/citeseer/ptb_0.05/seed_18/node_level/result.json",
      "efficiency/citeseer/ptb_0.05/seed_19/gb/result.json",
      "efficiency/citeseer/ptb_0.05/seed_19/kmeans/result.json",
      "efficiency/citeseer/ptb_0.05/seed_19/node_level/result.json",
      "efficiency/citeseer/ptb_0.1/seed_15/gb/result.json",
      "efficiency/citeseer/ptb_0.1/seed_15/kmeans/result.json",
      "efficiency/citeseer/ptb_0.1/seed_15/node_level/result.json",
      "efficiency/citeseer/ptb_0.1/seed_16/gb/result.json",
      "efficiency/citeseer/ptb_0.1/seed_16/kmeans/result.json",
      "efficiency/citeseer/ptb_0.1/seed_16/node_level/result.json",
      "efficiency/citeseer/ptb_0.1/seed_17/gb/result.json",
      "efficiency/citeseer/ptb_0.1/seed_17/kmeans/result.json",
      "efficiency/citeseer/ptb_0.1/seed_17/node_level/result.json",
      "efficiency/citeseer/ptb_0.1/seed_18/gb/result.json",
      "efficiency/citeseer/ptb_0.1/seed_18/kmeans/result.json",
      "efficiency/citeseer/ptb_0.1/seed_18/node_level/result.json",
      "efficiency/citeseer/ptb_0.1/seed_19/gb/result.json",
      "efficiency/citeseer/ptb_0.1/seed_19/kmeans/result.json",
      "efficiency/citeseer/ptb_0.1/seed_19/node_level/result.json",
      "efficiency/cora/ptb_0.01/seed_15/gb/result.json",
      "efficiency/cora/ptb_0.01/seed_15/kmeans/result.json",
      "efficiency/cora/ptb_0.01/seed_15/node_level/result.json",
      "efficiency/cora/ptb_0.01/seed_16/gb/result.json",
      "efficiency/cora/ptb_0.01/seed_16/kmeans/result.json",
      "efficiency/cora/ptb_0.01/seed_16/node_level/result.json",
      "efficiency/cora/ptb_0.01/seed_17/gb/result.json",
      "efficiency/cora/ptb_0.01/seed_17/kmeans/result.json",
      "efficiency/cora/ptb_0.01/seed_17/node_level/result.json",
      "efficiency/cora/ptb_0.01/seed_18/gb/result.json",
      "efficiency/cora/ptb_0.01/seed_18/kmeans/result.json",
      "efficiency/cora/ptb_0.01/seed_18/node_level/result.json",
      "efficiency/cora/ptb_0.01/seed_19/gb/result.json",
      "efficiency/cora/ptb_0.01/seed_19/kmeans/result.json",
      "efficiency/cora/ptb_0.01/seed_19/node_level/result.json",
      "efficiency/cora/ptb_0.05/seed_15/gb/result.json",
      "efficiency/cora/ptb_0.05/seed_15/kmeans/result.json",
      "efficiency/cora/ptb_0.05/seed_15/node_level/result.json",
      "efficiency/cora/ptb_0.05/seed_16/gb/result.json"
    ]
  },
  {
    "reason": "manuscript_mapping",
    "detail": "No supplied mapping from result cells to manuscript tables/figures",
    "paths": [
      "ae_ppr/citeseer/ptb_0.05/seed_15/ae_ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_15/ae_ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_15/ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_15/ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_16/ae_ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_16/ae_ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_16/ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_16/ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_17/ae_ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_17/ae_ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_17/ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_17/ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_18/ae_ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_18/ae_ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_18/ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_18/ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_19/ae_ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_19/ae_ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_19/ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_19/ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_15/ae_ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_15/ae_ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_15/ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_15/ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_16/ae_ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_16/ae_ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_16/ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_16/ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_17/ae_ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_17/ae_ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_17/ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_17/ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_17/seed_15/ae_ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_17/seed_15/ae_ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_17/seed_15/ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_17/seed_15/ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_18/ae_ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_18/ae_ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_18/ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_18/ppr_ppr/result.json",
      "hybrid/citeseer/ptb_0.05/seed_15/edge_only/result.json",
      "hybrid/citeseer/ptb_0.05/seed_15/feature_only/result.json",
      "hybrid/citeseer/ptb_0.05/seed_15/parallel/result.json",
      "hybrid/citeseer/ptb_0.05/seed_15/serial/result.json",
      "hybrid/citeseer/ptb_0.05/seed_16/edge_only/result.json",
      "hybrid/citeseer/ptb_0.05/seed_16/feature_only/result.json",
      "hybrid/citeseer/ptb_0.05/seed_16/parallel/result.json",
      "hybrid/citeseer/ptb_0.05/seed_16/serial/result.json",
      "hybrid/citeseer/ptb_0.05/seed_17/edge_only/result.json",
      "hybrid/citeseer/ptb_0.05/seed_17/feature_only/result.json",
      "hybrid/citeseer/ptb_0.05/seed_17/parallel/result.json",
      "hybrid/citeseer/ptb_0.05/seed_17/serial/result.json",
      "hybrid/citeseer/ptb_0.05/seed_18/edge_only/result.json",
      "hybrid/citeseer/ptb_0.05/seed_18/feature_only/result.json",
      "hybrid/citeseer/ptb_0.05/seed_18/parallel/result.json",
      "hybrid/citeseer/ptb_0.05/seed_18/serial/result.json",
      "hybrid/citeseer/ptb_0.05/seed_19/edge_only/result.json",
      "hybrid/citeseer/ptb_0.05/seed_19/feature_only/result.json",
      "hybrid/citeseer/ptb_0.05/seed_19/parallel/result.json",
      "hybrid/citeseer/ptb_0.05/seed_19/serial/result.json",
      "hybrid/citeseer/ptb_0.1/seed_15/edge_only/result.json",
      "hybrid/citeseer/ptb_0.1/seed_15/feature_only/result.json",
      "hybrid/citeseer/ptb_0.1/seed_15/parallel/result.json",
      "hybrid/citeseer/ptb_0.1/seed_15/serial/result.json",
      "hybrid/citeseer/ptb_0.1/seed_16/edge_only/result.json",
      "hybrid/citeseer/ptb_0.1/seed_16/feature_only/result.json",
      "hybrid/citeseer/ptb_0.1/seed_16/parallel/result.json",
      "hybrid/citeseer/ptb_0.1/seed_16/serial/result.json",
      "hybrid/citeseer/ptb_0.1/seed_17/edge_only/result.json",
      "hybrid/citeseer/ptb_0.1/seed_17/feature_only/result.json",
      "hybrid/citeseer/ptb_0.1/seed_17/parallel/result.json",
      "hybrid/citeseer/ptb_0.1/seed_17/serial/result.json",
      "hybrid/citeseer/ptb_0.1/seed_18/edge_only/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_15/gb/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_15/kmeans/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_15/node_level/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_15/random/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_16/gb/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_16/kmeans/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_16/node_level/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_16/random/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_17/gb/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_17/kmeans/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_17/node_level/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_17/random/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_18/gb/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_18/kmeans/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_18/node_level/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_18/random/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_19/gb/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_19/kmeans/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_19/node_level/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_19/random/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_15/gb/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_15/kmeans/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_15/node_level/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_15/random/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_16/gb/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_16/kmeans/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_16/node_level/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_16/random/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_17/gb/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_17/kmeans/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_17/node_level/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_17/random/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_18/gb/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_18/kmeans/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_18/node_level/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_18/random/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_19/gb/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_19/kmeans/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_19/node_level/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_19/random/result.json",
      "efficiency/citeseer/ptb_0.01/seed_15/gb/result.json",
      "efficiency/citeseer/ptb_0.01/seed_15/kmeans/result.json",
      "efficiency/citeseer/ptb_0.01/seed_15/node_level/result.json",
      "efficiency/citeseer/ptb_0.01/seed_16/gb/result.json",
      "efficiency/citeseer/ptb_0.01/seed_16/kmeans/result.json",
      "efficiency/citeseer/ptb_0.01/seed_16/node_level/result.json",
      "efficiency/citeseer/ptb_0.01/seed_17/gb/result.json",
      "efficiency/citeseer/ptb_0.01/seed_17/kmeans/result.json",
      "efficiency/citeseer/ptb_0.01/seed_17/node_level/result.json",
      "efficiency/citeseer/ptb_0.01/seed_18/gb/result.json",
      "efficiency/citeseer/ptb_0.01/seed_18/kmeans/result.json",
      "efficiency/citeseer/ptb_0.01/seed_18/node_level/result.json",
      "efficiency/citeseer/ptb_0.01/seed_19/gb/result.json",
      "efficiency/citeseer/ptb_0.01/seed_19/kmeans/result.json",
      "efficiency/citeseer/ptb_0.01/seed_19/node_level/result.json",
      "efficiency/citeseer/ptb_0.05/seed_15/gb/result.json",
      "efficiency/citeseer/ptb_0.05/seed_15/kmeans/result.json",
      "efficiency/citeseer/ptb_0.05/seed_15/node_level/result.json",
      "efficiency/citeseer/ptb_0.05/seed_16/gb/result.json",
      "efficiency/citeseer/ptb_0.05/seed_16/kmeans/result.json",
      "efficiency/citeseer/ptb_0.05/seed_16/node_level/result.json",
      "efficiency/citeseer/ptb_0.05/seed_17/gb/result.json",
      "efficiency/citeseer/ptb_0.05/seed_17/kmeans/result.json",
      "efficiency/citeseer/ptb_0.05/seed_17/node_level/result.json",
      "efficiency/citeseer/ptb_0.05/seed_18/gb/result.json",
      "efficiency/citeseer/ptb_0.05/seed_18/kmeans/result.json",
      "efficiency/citeseer/ptb_0.05/seed_18/node_level/result.json",
      "efficiency/citeseer/ptb_0.05/seed_19/gb/result.json",
      "efficiency/citeseer/ptb_0.05/seed_19/kmeans/result.json",
      "efficiency/citeseer/ptb_0.05/seed_19/node_level/result.json",
      "efficiency/citeseer/ptb_0.1/seed_15/gb/result.json",
      "efficiency/citeseer/ptb_0.1/seed_15/kmeans/result.json",
      "efficiency/citeseer/ptb_0.1/seed_15/node_level/result.json",
      "efficiency/citeseer/ptb_0.1/seed_16/gb/result.json",
      "efficiency/citeseer/ptb_0.1/seed_16/kmeans/result.json",
      "efficiency/citeseer/ptb_0.1/seed_16/node_level/result.json",
      "efficiency/citeseer/ptb_0.1/seed_17/gb/result.json",
      "efficiency/citeseer/ptb_0.1/seed_17/kmeans/result.json",
      "efficiency/citeseer/ptb_0.1/seed_17/node_level/result.json",
      "efficiency/citeseer/ptb_0.1/seed_18/gb/result.json",
      "efficiency/citeseer/ptb_0.1/seed_18/kmeans/result.json",
      "efficiency/citeseer/ptb_0.1/seed_18/node_level/result.json",
      "efficiency/citeseer/ptb_0.1/seed_19/gb/result.json",
      "efficiency/citeseer/ptb_0.1/seed_19/kmeans/result.json",
      "efficiency/citeseer/ptb_0.1/seed_19/node_level/result.json",
      "efficiency/cora/ptb_0.01/seed_15/gb/result.json",
      "efficiency/cora/ptb_0.01/seed_15/kmeans/result.json",
      "efficiency/cora/ptb_0.01/seed_15/node_level/result.json",
      "efficiency/cora/ptb_0.01/seed_16/gb/result.json",
      "efficiency/cora/ptb_0.01/seed_16/kmeans/result.json",
      "efficiency/cora/ptb_0.01/seed_16/node_level/result.json",
      "efficiency/cora/ptb_0.01/seed_17/gb/result.json",
      "efficiency/cora/ptb_0.01/seed_17/kmeans/result.json",
      "efficiency/cora/ptb_0.01/seed_17/node_level/result.json",
      "efficiency/cora/ptb_0.01/seed_18/gb/result.json",
      "efficiency/cora/ptb_0.01/seed_18/kmeans/result.json",
      "efficiency/cora/ptb_0.01/seed_18/node_level/result.json",
      "efficiency/cora/ptb_0.01/seed_19/gb/result.json",
      "efficiency/cora/ptb_0.01/seed_19/kmeans/result.json",
      "efficiency/cora/ptb_0.01/seed_19/node_level/result.json",
      "efficiency/cora/ptb_0.05/seed_15/gb/result.json",
      "efficiency/cora/ptb_0.05/seed_15/kmeans/result.json",
      "efficiency/cora/ptb_0.05/seed_15/node_level/result.json",
      "efficiency/cora/ptb_0.05/seed_16/gb/result.json"
    ]
  },
  {
    "reason": "paired_artifact_provenance",
    "detail": "Graph/split/features, trajectory, aligned predictions and RNG cannot be verified from summaries",
    "paths": [
      "ae_ppr/citeseer/ptb_0.05/seed_15/ae_ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_15/ae_ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_15/ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_15/ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_16/ae_ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_16/ae_ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_16/ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_16/ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_17/ae_ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_17/ae_ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_17/ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_17/ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_18/ae_ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_18/ae_ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_18/ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_18/ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_19/ae_ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_19/ae_ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_19/ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.05/seed_19/ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_15/ae_ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_15/ae_ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_15/ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_15/ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_16/ae_ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_16/ae_ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_16/ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_16/ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_17/ae_ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_17/ae_ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_17/ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_17/ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_17/seed_15/ae_ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_17/seed_15/ae_ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_17/seed_15/ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_17/seed_15/ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_18/ae_ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_18/ae_ppr_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_18/ppr_ae_ppr/result.json",
      "ae_ppr/citeseer/ptb_0.1/seed_18/ppr_ppr/result.json",
      "hybrid/citeseer/ptb_0.05/seed_15/edge_only/result.json",
      "hybrid/citeseer/ptb_0.05/seed_15/feature_only/result.json",
      "hybrid/citeseer/ptb_0.05/seed_15/parallel/result.json",
      "hybrid/citeseer/ptb_0.05/seed_15/serial/result.json",
      "hybrid/citeseer/ptb_0.05/seed_16/edge_only/result.json",
      "hybrid/citeseer/ptb_0.05/seed_16/feature_only/result.json",
      "hybrid/citeseer/ptb_0.05/seed_16/parallel/result.json",
      "hybrid/citeseer/ptb_0.05/seed_16/serial/result.json",
      "hybrid/citeseer/ptb_0.05/seed_17/edge_only/result.json",
      "hybrid/citeseer/ptb_0.05/seed_17/feature_only/result.json",
      "hybrid/citeseer/ptb_0.05/seed_17/parallel/result.json",
      "hybrid/citeseer/ptb_0.05/seed_17/serial/result.json",
      "hybrid/citeseer/ptb_0.05/seed_18/edge_only/result.json",
      "hybrid/citeseer/ptb_0.05/seed_18/feature_only/result.json",
      "hybrid/citeseer/ptb_0.05/seed_18/parallel/result.json",
      "hybrid/citeseer/ptb_0.05/seed_18/serial/result.json",
      "hybrid/citeseer/ptb_0.05/seed_19/edge_only/result.json",
      "hybrid/citeseer/ptb_0.05/seed_19/feature_only/result.json",
      "hybrid/citeseer/ptb_0.05/seed_19/parallel/result.json",
      "hybrid/citeseer/ptb_0.05/seed_19/serial/result.json",
      "hybrid/citeseer/ptb_0.1/seed_15/edge_only/result.json",
      "hybrid/citeseer/ptb_0.1/seed_15/feature_only/result.json",
      "hybrid/citeseer/ptb_0.1/seed_15/parallel/result.json",
      "hybrid/citeseer/ptb_0.1/seed_15/serial/result.json",
      "hybrid/citeseer/ptb_0.1/seed_16/edge_only/result.json",
      "hybrid/citeseer/ptb_0.1/seed_16/feature_only/result.json",
      "hybrid/citeseer/ptb_0.1/seed_16/parallel/result.json",
      "hybrid/citeseer/ptb_0.1/seed_16/serial/result.json",
      "hybrid/citeseer/ptb_0.1/seed_17/edge_only/result.json",
      "hybrid/citeseer/ptb_0.1/seed_17/feature_only/result.json",
      "hybrid/citeseer/ptb_0.1/seed_17/parallel/result.json",
      "hybrid/citeseer/ptb_0.1/seed_17/serial/result.json",
      "hybrid/citeseer/ptb_0.1/seed_18/edge_only/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_15/gb/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_15/kmeans/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_15/node_level/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_15/random/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_16/gb/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_16/kmeans/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_16/node_level/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_16/random/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_17/gb/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_17/kmeans/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_17/node_level/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_17/random/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_18/gb/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_18/kmeans/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_18/node_level/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_18/random/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_19/gb/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_19/kmeans/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_19/node_level/result.json",
      "gb_ablation/citeseer/ptb_0.05/seed_19/random/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_15/gb/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_15/kmeans/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_15/node_level/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_15/random/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_16/gb/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_16/kmeans/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_16/node_level/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_16/random/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_17/gb/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_17/kmeans/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_17/node_level/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_17/random/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_18/gb/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_18/kmeans/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_18/node_level/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_18/random/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_19/gb/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_19/kmeans/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_19/node_level/result.json",
      "gb_ablation/citeseer/ptb_0.1/seed_19/random/result.json",
      "efficiency/citeseer/ptb_0.01/seed_15/gb/result.json",
      "efficiency/citeseer/ptb_0.01/seed_15/kmeans/result.json",
      "efficiency/citeseer/ptb_0.01/seed_15/node_level/result.json",
      "efficiency/citeseer/ptb_0.01/seed_16/gb/result.json",
      "efficiency/citeseer/ptb_0.01/seed_16/kmeans/result.json",
      "efficiency/citeseer/ptb_0.01/seed_16/node_level/result.json",
      "efficiency/citeseer/ptb_0.01/seed_17/gb/result.json",
      "efficiency/citeseer/ptb_0.01/seed_17/kmeans/result.json",
      "efficiency/citeseer/ptb_0.01/seed_17/node_level/result.json",
      "efficiency/citeseer/ptb_0.01/seed_18/gb/result.json",
      "efficiency/citeseer/ptb_0.01/seed_18/kmeans/result.json",
      "efficiency/citeseer/ptb_0.01/seed_18/node_level/result.json",
      "efficiency/citeseer/ptb_0.01/seed_19/gb/result.json",
      "efficiency/citeseer/ptb_0.01/seed_19/kmeans/result.json",
      "efficiency/citeseer/ptb_0.01/seed_19/node_level/result.json",
      "efficiency/citeseer/ptb_0.05/seed_15/gb/result.json",
      "efficiency/citeseer/ptb_0.05/seed_15/kmeans/result.json",
      "efficiency/citeseer/ptb_0.05/seed_15/node_level/result.json",
      "efficiency/citeseer/ptb_0.05/seed_16/gb/result.json",
      "efficiency/citeseer/ptb_0.05/seed_16/kmeans/result.json",
      "efficiency/citeseer/ptb_0.05/seed_16/node_level/result.json",
      "efficiency/citeseer/ptb_0.05/seed_17/gb/result.json",
      "efficiency/citeseer/ptb_0.05/seed_17/kmeans/result.json",
      "efficiency/citeseer/ptb_0.05/seed_17/node_level/result.json",
      "efficiency/citeseer/ptb_0.05/seed_18/gb/result.json",
      "efficiency/citeseer/ptb_0.05/seed_18/kmeans/result.json",
      "efficiency/citeseer/ptb_0.05/seed_18/node_level/result.json",
      "efficiency/citeseer/ptb_0.05/seed_19/gb/result.json",
      "efficiency/citeseer/ptb_0.05/seed_19/kmeans/result.json",
      "efficiency/citeseer/ptb_0.05/seed_19/node_level/result.json",
      "efficiency/citeseer/ptb_0.1/seed_15/gb/result.json",
      "efficiency/citeseer/ptb_0.1/seed_15/kmeans/result.json",
      "efficiency/citeseer/ptb_0.1/seed_15/node_level/result.json",
      "efficiency/citeseer/ptb_0.1/seed_16/gb/result.json",
      "efficiency/citeseer/ptb_0.1/seed_16/kmeans/result.json",
      "efficiency/citeseer/ptb_0.1/seed_16/node_level/result.json",
      "efficiency/citeseer/ptb_0.1/seed_17/gb/result.json",
      "efficiency/citeseer/ptb_0.1/seed_17/kmeans/result.json",
      "efficiency/citeseer/ptb_0.1/seed_17/node_level/result.json",
      "efficiency/citeseer/ptb_0.1/seed_18/gb/result.json",
      "efficiency/citeseer/ptb_0.1/seed_18/kmeans/result.json",
      "efficiency/citeseer/ptb_0.1/seed_18/node_level/result.json",
      "efficiency/citeseer/ptb_0.1/seed_19/gb/result.json",
      "efficiency/citeseer/ptb_0.1/seed_19/kmeans/result.json",
      "efficiency/citeseer/ptb_0.1/seed_19/node_level/result.json",
      "efficiency/cora/ptb_0.01/seed_15/gb/result.json",
      "efficiency/cora/ptb_0.01/seed_15/kmeans/result.json",
      "efficiency/cora/ptb_0.01/seed_15/node_level/result.json",
      "efficiency/cora/ptb_0.01/seed_16/gb/result.json",
      "efficiency/cora/ptb_0.01/seed_16/kmeans/result.json",
      "efficiency/cora/ptb_0.01/seed_16/node_level/result.json",
      "efficiency/cora/ptb_0.01/seed_17/gb/result.json",
      "efficiency/cora/ptb_0.01/seed_17/kmeans/result.json",
      "efficiency/cora/ptb_0.01/seed_17/node_level/result.json",
      "efficiency/cora/ptb_0.01/seed_18/gb/result.json",
      "efficiency/cora/ptb_0.01/seed_18/kmeans/result.json",
      "efficiency/cora/ptb_0.01/seed_18/node_level/result.json",
      "efficiency/cora/ptb_0.01/seed_19/gb/result.json",
      "efficiency/cora/ptb_0.01/seed_19/kmeans/result.json",
      "efficiency/cora/ptb_0.01/seed_19/node_level/result.json",
      "efficiency/cora/ptb_0.05/seed_15/gb/result.json",
      "efficiency/cora/ptb_0.05/seed_15/kmeans/result.json",
      "efficiency/cora/ptb_0.05/seed_15/node_level/result.json",
      "efficiency/cora/ptb_0.05/seed_16/gb/result.json"
    ]
  }
]
```

## Per-record inventory

Complete sanitized commands, configuration details and evidence are retained in `audit.json`; credential values are represented by `<redacted>`.

| Record | Global source | Global alpha / iterations | Local alpha / iterations | Issues |
|---|---|---|---|---|
| `ae_ppr/citeseer/ptb_0.05/seed_15/ae_ppr_ae_ppr/result.json` | degree | 0.15 / 30 | 0.28 / 60 | none beyond historical provenance |
| `ae_ppr/citeseer/ptb_0.05/seed_15/ae_ppr_ppr/result.json` | degree | 0.15 / 30 | 0.28 / 60 | none beyond historical provenance |
| `ae_ppr/citeseer/ptb_0.05/seed_15/ppr_ae_ppr/result.json` | degree | 0.15 / 30 | 0.28 / 60 | none beyond historical provenance |
| `ae_ppr/citeseer/ptb_0.05/seed_15/ppr_ppr/result.json` | degree | 0.15 / 30 | 0.28 / 60 | none beyond historical provenance |
| `ae_ppr/citeseer/ptb_0.05/seed_16/ae_ppr_ae_ppr/result.json` | degree | 0.15 / 30 | 0.28 / 60 | none beyond historical provenance |
| `ae_ppr/citeseer/ptb_0.05/seed_16/ae_ppr_ppr/result.json` | degree | 0.15 / 30 | 0.28 / 60 | none beyond historical provenance |
| `ae_ppr/citeseer/ptb_0.05/seed_16/ppr_ae_ppr/result.json` | degree | 0.15 / 30 | 0.28 / 60 | none beyond historical provenance |
| `ae_ppr/citeseer/ptb_0.05/seed_16/ppr_ppr/result.json` | degree | 0.15 / 30 | 0.28 / 60 | none beyond historical provenance |
| `ae_ppr/citeseer/ptb_0.05/seed_17/ae_ppr_ae_ppr/result.json` | degree | 0.15 / 30 | 0.28 / 60 | none beyond historical provenance |
| `ae_ppr/citeseer/ptb_0.05/seed_17/ae_ppr_ppr/result.json` | degree | 0.15 / 30 | 0.28 / 60 | none beyond historical provenance |
| `ae_ppr/citeseer/ptb_0.05/seed_17/ppr_ae_ppr/result.json` | degree | 0.15 / 30 | 0.28 / 60 | none beyond historical provenance |
| `ae_ppr/citeseer/ptb_0.05/seed_17/ppr_ppr/result.json` | degree | 0.15 / 30 | 0.28 / 60 | none beyond historical provenance |
| `ae_ppr/citeseer/ptb_0.05/seed_18/ae_ppr_ae_ppr/result.json` | degree | 0.15 / 30 | 0.28 / 60 | none beyond historical provenance |
| `ae_ppr/citeseer/ptb_0.05/seed_18/ae_ppr_ppr/result.json` | degree | 0.15 / 30 | 0.28 / 60 | none beyond historical provenance |
| `ae_ppr/citeseer/ptb_0.05/seed_18/ppr_ae_ppr/result.json` | degree | 0.15 / 30 | 0.28 / 60 | none beyond historical provenance |
| `ae_ppr/citeseer/ptb_0.05/seed_18/ppr_ppr/result.json` | degree | 0.15 / 30 | 0.28 / 60 | none beyond historical provenance |
| `ae_ppr/citeseer/ptb_0.05/seed_19/ae_ppr_ae_ppr/result.json` | degree | 0.15 / 30 | 0.28 / 60 | none beyond historical provenance |
| `ae_ppr/citeseer/ptb_0.05/seed_19/ae_ppr_ppr/result.json` | degree | 0.15 / 30 | 0.28 / 60 | none beyond historical provenance |
| `ae_ppr/citeseer/ptb_0.05/seed_19/ppr_ae_ppr/result.json` | degree | 0.15 / 30 | 0.28 / 60 | none beyond historical provenance |
| `ae_ppr/citeseer/ptb_0.05/seed_19/ppr_ppr/result.json` | degree | 0.15 / 30 | 0.28 / 60 | none beyond historical provenance |
| `ae_ppr/citeseer/ptb_0.1/seed_15/ae_ppr_ae_ppr/result.json` | degree | 0.15 / 30 | 0.28 / 60 | misclassification_disagreement, drop_disagreement, duplicate_cell |
| `ae_ppr/citeseer/ptb_0.1/seed_15/ae_ppr_ppr/result.json` | degree | 0.15 / 30 | 0.28 / 60 | duplicate_cell |
| `ae_ppr/citeseer/ptb_0.1/seed_15/ppr_ae_ppr/result.json` | degree | 0.15 / 30 | 0.28 / 60 | duplicate_cell |
| `ae_ppr/citeseer/ptb_0.1/seed_15/ppr_ppr/result.json` | degree | 0.15 / 30 | 0.28 / 60 | duplicate_cell |
| `ae_ppr/citeseer/ptb_0.1/seed_16/ae_ppr_ae_ppr/result.json` | degree | 0.15 / 30 | 0.28 / 60 | cell_mapping_conflict, misclassification_disagreement, drop_disagreement |
| `ae_ppr/citeseer/ptb_0.1/seed_16/ae_ppr_ppr/result.json` | degree | 0.15 / 30 | 0.28 / 60 | cell_mapping_conflict |
| `ae_ppr/citeseer/ptb_0.1/seed_16/ppr_ae_ppr/result.json` | degree | 0.15 / 30 | 0.28 / 60 | cell_mapping_conflict |
| `ae_ppr/citeseer/ptb_0.1/seed_16/ppr_ppr/result.json` | degree | 0.15 / 30 | 0.28 / 60 | cell_mapping_conflict |
| `ae_ppr/citeseer/ptb_0.1/seed_17/ae_ppr_ae_ppr/result.json` | degree | 0.15 / 30 | 0.28 / 60 | cell_mapping_conflict, misclassification_disagreement, drop_disagreement |
| `ae_ppr/citeseer/ptb_0.1/seed_17/ae_ppr_ppr/result.json` | degree | 0.15 / 30 | 0.28 / 60 | cell_mapping_conflict |
| `ae_ppr/citeseer/ptb_0.1/seed_17/ppr_ae_ppr/result.json` | degree | 0.15 / 30 | 0.28 / 60 | cell_mapping_conflict |
| `ae_ppr/citeseer/ptb_0.1/seed_17/ppr_ppr/result.json` | degree | 0.15 / 30 | 0.28 / 60 | cell_mapping_conflict |
| `ae_ppr/citeseer/ptb_0.1/seed_17/seed_15/ae_ppr_ae_ppr/result.json` | degree | 0.15 / 30 | 0.28 / 60 | path_mapping_unresolved, misclassification_disagreement, drop_disagreement, duplicate_cell |
| `ae_ppr/citeseer/ptb_0.1/seed_17/seed_15/ae_ppr_ppr/result.json` | degree | 0.15 / 30 | 0.28 / 60 | path_mapping_unresolved, duplicate_cell |
| `ae_ppr/citeseer/ptb_0.1/seed_17/seed_15/ppr_ae_ppr/result.json` | degree | 0.15 / 30 | 0.28 / 60 | path_mapping_unresolved, duplicate_cell |
| `ae_ppr/citeseer/ptb_0.1/seed_17/seed_15/ppr_ppr/result.json` | degree | 0.15 / 30 | 0.28 / 60 | path_mapping_unresolved, duplicate_cell |
| `ae_ppr/citeseer/ptb_0.1/seed_18/ae_ppr_ae_ppr/result.json` | degree | 0.15 / 30 | 0.28 / 60 | cell_mapping_conflict, misclassification_disagreement, drop_disagreement |
| `ae_ppr/citeseer/ptb_0.1/seed_18/ae_ppr_ppr/result.json` | degree | 0.15 / 30 | 0.28 / 60 | cell_mapping_conflict |
| `ae_ppr/citeseer/ptb_0.1/seed_18/ppr_ae_ppr/result.json` | degree | 0.15 / 30 | 0.28 / 60 | cell_mapping_conflict |
| `ae_ppr/citeseer/ptb_0.1/seed_18/ppr_ppr/result.json` | degree | 0.15 / 30 | 0.28 / 60 | cell_mapping_conflict |
| `hybrid/citeseer/ptb_0.05/seed_15/edge_only/result.json` | degree | 0.15 / 30 | None / None | none beyond historical provenance |
| `hybrid/citeseer/ptb_0.05/seed_15/feature_only/result.json` | degree | 0.15 / 30 | 0.28 / 60 | none beyond historical provenance |
| `hybrid/citeseer/ptb_0.05/seed_15/parallel/result.json` | degree | 0.15 / 30 | 0.28 / 60 | none beyond historical provenance |
| `hybrid/citeseer/ptb_0.05/seed_15/serial/result.json` | degree | 0.15 / 30 | 0.28 / 60 | none beyond historical provenance |
| `hybrid/citeseer/ptb_0.05/seed_16/edge_only/result.json` | degree | 0.15 / 30 | None / None | none beyond historical provenance |
| `hybrid/citeseer/ptb_0.05/seed_16/feature_only/result.json` | degree | 0.15 / 30 | 0.28 / 60 | none beyond historical provenance |
| `hybrid/citeseer/ptb_0.05/seed_16/parallel/result.json` | degree | 0.15 / 30 | 0.28 / 60 | none beyond historical provenance |
| `hybrid/citeseer/ptb_0.05/seed_16/serial/result.json` | degree | 0.15 / 30 | 0.28 / 60 | none beyond historical provenance |
| `hybrid/citeseer/ptb_0.05/seed_17/edge_only/result.json` | degree | 0.15 / 30 | None / None | none beyond historical provenance |
| `hybrid/citeseer/ptb_0.05/seed_17/feature_only/result.json` | degree | 0.15 / 30 | 0.28 / 60 | none beyond historical provenance |
| `hybrid/citeseer/ptb_0.05/seed_17/parallel/result.json` | degree | 0.15 / 30 | 0.28 / 60 | none beyond historical provenance |
| `hybrid/citeseer/ptb_0.05/seed_17/serial/result.json` | degree | 0.15 / 30 | 0.28 / 60 | none beyond historical provenance |
| `hybrid/citeseer/ptb_0.05/seed_18/edge_only/result.json` | degree | 0.15 / 30 | None / None | none beyond historical provenance |
| `hybrid/citeseer/ptb_0.05/seed_18/feature_only/result.json` | degree | 0.15 / 30 | 0.28 / 60 | none beyond historical provenance |
| `hybrid/citeseer/ptb_0.05/seed_18/parallel/result.json` | degree | 0.15 / 30 | 0.28 / 60 | none beyond historical provenance |
| `hybrid/citeseer/ptb_0.05/seed_18/serial/result.json` | degree | 0.15 / 30 | 0.28 / 60 | none beyond historical provenance |
| `hybrid/citeseer/ptb_0.05/seed_19/edge_only/result.json` | degree | 0.15 / 30 | None / None | none beyond historical provenance |
| `hybrid/citeseer/ptb_0.05/seed_19/feature_only/result.json` | degree | 0.15 / 30 | 0.28 / 60 | none beyond historical provenance |
| `hybrid/citeseer/ptb_0.05/seed_19/parallel/result.json` | degree | 0.15 / 30 | 0.28 / 60 | none beyond historical provenance |
| `hybrid/citeseer/ptb_0.05/seed_19/serial/result.json` | degree | 0.15 / 30 | 0.28 / 60 | none beyond historical provenance |
| `hybrid/citeseer/ptb_0.1/seed_15/edge_only/result.json` | degree | 0.15 / 30 | None / None | none beyond historical provenance |
| `hybrid/citeseer/ptb_0.1/seed_15/feature_only/result.json` | degree | 0.15 / 30 | 0.28 / 60 | none beyond historical provenance |
| `hybrid/citeseer/ptb_0.1/seed_15/parallel/result.json` | degree | 0.15 / 30 | 0.28 / 60 | none beyond historical provenance |
| `hybrid/citeseer/ptb_0.1/seed_15/serial/result.json` | degree | 0.15 / 30 | 0.28 / 60 | none beyond historical provenance |
| `hybrid/citeseer/ptb_0.1/seed_16/edge_only/result.json` | degree | 0.15 / 30 | None / None | none beyond historical provenance |
| `hybrid/citeseer/ptb_0.1/seed_16/feature_only/result.json` | degree | 0.15 / 30 | 0.28 / 60 | none beyond historical provenance |
| `hybrid/citeseer/ptb_0.1/seed_16/parallel/result.json` | degree | 0.15 / 30 | 0.28 / 60 | none beyond historical provenance |
| `hybrid/citeseer/ptb_0.1/seed_16/serial/result.json` | degree | 0.15 / 30 | 0.28 / 60 | none beyond historical provenance |
| `hybrid/citeseer/ptb_0.1/seed_17/edge_only/result.json` | degree | 0.15 / 30 | None / None | none beyond historical provenance |
| `hybrid/citeseer/ptb_0.1/seed_17/feature_only/result.json` | degree | 0.15 / 30 | 0.28 / 60 | none beyond historical provenance |
| `hybrid/citeseer/ptb_0.1/seed_17/parallel/result.json` | degree | 0.15 / 30 | 0.28 / 60 | none beyond historical provenance |
| `hybrid/citeseer/ptb_0.1/seed_17/serial/result.json` | degree | 0.15 / 30 | 0.28 / 60 | none beyond historical provenance |
| `hybrid/citeseer/ptb_0.1/seed_18/edge_only/result.json` | degree | 0.15 / 30 | None / None | none beyond historical provenance |
| `gb_ablation/citeseer/ptb_0.05/seed_15/gb/result.json` | degree | 0.15 / 30 | None / None | none beyond historical provenance |
| `gb_ablation/citeseer/ptb_0.05/seed_15/kmeans/result.json` | degree | 0.15 / 30 | None / None | none beyond historical provenance |
| `gb_ablation/citeseer/ptb_0.05/seed_15/node_level/result.json` | degree | 0.15 / 30 | None / None | none beyond historical provenance |
| `gb_ablation/citeseer/ptb_0.05/seed_15/random/result.json` | degree | 0.15 / 30 | None / None | none beyond historical provenance |
| `gb_ablation/citeseer/ptb_0.05/seed_16/gb/result.json` | degree | 0.15 / 30 | None / None | none beyond historical provenance |
| `gb_ablation/citeseer/ptb_0.05/seed_16/kmeans/result.json` | degree | 0.15 / 30 | None / None | none beyond historical provenance |
| `gb_ablation/citeseer/ptb_0.05/seed_16/node_level/result.json` | degree | 0.15 / 30 | None / None | none beyond historical provenance |
| `gb_ablation/citeseer/ptb_0.05/seed_16/random/result.json` | degree | 0.15 / 30 | None / None | none beyond historical provenance |
| `gb_ablation/citeseer/ptb_0.05/seed_17/gb/result.json` | degree | 0.15 / 30 | None / None | none beyond historical provenance |
| `gb_ablation/citeseer/ptb_0.05/seed_17/kmeans/result.json` | degree | 0.15 / 30 | None / None | none beyond historical provenance |
| `gb_ablation/citeseer/ptb_0.05/seed_17/node_level/result.json` | degree | 0.15 / 30 | None / None | none beyond historical provenance |
| `gb_ablation/citeseer/ptb_0.05/seed_17/random/result.json` | degree | 0.15 / 30 | None / None | none beyond historical provenance |
| `gb_ablation/citeseer/ptb_0.05/seed_18/gb/result.json` | degree | 0.15 / 30 | None / None | none beyond historical provenance |
| `gb_ablation/citeseer/ptb_0.05/seed_18/kmeans/result.json` | degree | 0.15 / 30 | None / None | none beyond historical provenance |
| `gb_ablation/citeseer/ptb_0.05/seed_18/node_level/result.json` | degree | 0.15 / 30 | None / None | none beyond historical provenance |
| `gb_ablation/citeseer/ptb_0.05/seed_18/random/result.json` | degree | 0.15 / 30 | None / None | none beyond historical provenance |
| `gb_ablation/citeseer/ptb_0.05/seed_19/gb/result.json` | degree | 0.15 / 30 | None / None | none beyond historical provenance |
| `gb_ablation/citeseer/ptb_0.05/seed_19/kmeans/result.json` | degree | 0.15 / 30 | None / None | none beyond historical provenance |
| `gb_ablation/citeseer/ptb_0.05/seed_19/node_level/result.json` | degree | 0.15 / 30 | None / None | none beyond historical provenance |
| `gb_ablation/citeseer/ptb_0.05/seed_19/random/result.json` | degree | 0.15 / 30 | None / None | none beyond historical provenance |
| `gb_ablation/citeseer/ptb_0.1/seed_15/gb/result.json` | degree | 0.15 / 30 | None / None | none beyond historical provenance |
| `gb_ablation/citeseer/ptb_0.1/seed_15/kmeans/result.json` | degree | 0.15 / 30 | None / None | none beyond historical provenance |
| `gb_ablation/citeseer/ptb_0.1/seed_15/node_level/result.json` | degree | 0.15 / 30 | None / None | none beyond historical provenance |
| `gb_ablation/citeseer/ptb_0.1/seed_15/random/result.json` | degree | 0.15 / 30 | None / None | none beyond historical provenance |
| `gb_ablation/citeseer/ptb_0.1/seed_16/gb/result.json` | degree | 0.15 / 30 | None / None | none beyond historical provenance |
| `gb_ablation/citeseer/ptb_0.1/seed_16/kmeans/result.json` | degree | 0.15 / 30 | None / None | none beyond historical provenance |
| `gb_ablation/citeseer/ptb_0.1/seed_16/node_level/result.json` | degree | 0.15 / 30 | None / None | none beyond historical provenance |
| `gb_ablation/citeseer/ptb_0.1/seed_16/random/result.json` | degree | 0.15 / 30 | None / None | none beyond historical provenance |
| `gb_ablation/citeseer/ptb_0.1/seed_17/gb/result.json` | degree | 0.15 / 30 | None / None | none beyond historical provenance |
| `gb_ablation/citeseer/ptb_0.1/seed_17/kmeans/result.json` | degree | 0.15 / 30 | None / None | none beyond historical provenance |
| `gb_ablation/citeseer/ptb_0.1/seed_17/node_level/result.json` | degree | 0.15 / 30 | None / None | none beyond historical provenance |
| `gb_ablation/citeseer/ptb_0.1/seed_17/random/result.json` | degree | 0.15 / 30 | None / None | none beyond historical provenance |
| `gb_ablation/citeseer/ptb_0.1/seed_18/gb/result.json` | degree | 0.15 / 30 | None / None | none beyond historical provenance |
| `gb_ablation/citeseer/ptb_0.1/seed_18/kmeans/result.json` | degree | 0.15 / 30 | None / None | none beyond historical provenance |
| `gb_ablation/citeseer/ptb_0.1/seed_18/node_level/result.json` | degree | 0.15 / 30 | None / None | none beyond historical provenance |
| `gb_ablation/citeseer/ptb_0.1/seed_18/random/result.json` | degree | 0.15 / 30 | None / None | none beyond historical provenance |
| `gb_ablation/citeseer/ptb_0.1/seed_19/gb/result.json` | degree | 0.15 / 30 | None / None | none beyond historical provenance |
| `gb_ablation/citeseer/ptb_0.1/seed_19/kmeans/result.json` | degree | 0.15 / 30 | None / None | none beyond historical provenance |
| `gb_ablation/citeseer/ptb_0.1/seed_19/node_level/result.json` | degree | 0.15 / 30 | None / None | none beyond historical provenance |
| `gb_ablation/citeseer/ptb_0.1/seed_19/random/result.json` | degree | 0.15 / 30 | None / None | none beyond historical provenance |
| `efficiency/citeseer/ptb_0.01/seed_15/gb/result.json` | degree | 0.08 / 120 | 0.26 / 30 | none beyond historical provenance |
| `efficiency/citeseer/ptb_0.01/seed_15/kmeans/result.json` | degree | 0.08 / 120 | 0.26 / 30 | none beyond historical provenance |
| `efficiency/citeseer/ptb_0.01/seed_15/node_level/result.json` | degree | 0.08 / 120 | 0.26 / 30 | none beyond historical provenance |
| `efficiency/citeseer/ptb_0.01/seed_16/gb/result.json` | degree | 0.08 / 120 | 0.26 / 30 | none beyond historical provenance |
| `efficiency/citeseer/ptb_0.01/seed_16/kmeans/result.json` | degree | 0.08 / 120 | 0.26 / 30 | none beyond historical provenance |
| `efficiency/citeseer/ptb_0.01/seed_16/node_level/result.json` | degree | 0.08 / 120 | 0.26 / 30 | none beyond historical provenance |
| `efficiency/citeseer/ptb_0.01/seed_17/gb/result.json` | degree | 0.08 / 120 | 0.26 / 30 | none beyond historical provenance |
| `efficiency/citeseer/ptb_0.01/seed_17/kmeans/result.json` | degree | 0.08 / 120 | 0.26 / 30 | none beyond historical provenance |
| `efficiency/citeseer/ptb_0.01/seed_17/node_level/result.json` | degree | 0.08 / 120 | 0.26 / 30 | none beyond historical provenance |
| `efficiency/citeseer/ptb_0.01/seed_18/gb/result.json` | degree | 0.08 / 120 | 0.26 / 30 | none beyond historical provenance |
| `efficiency/citeseer/ptb_0.01/seed_18/kmeans/result.json` | degree | 0.08 / 120 | 0.26 / 30 | none beyond historical provenance |
| `efficiency/citeseer/ptb_0.01/seed_18/node_level/result.json` | degree | 0.08 / 120 | 0.26 / 30 | none beyond historical provenance |
| `efficiency/citeseer/ptb_0.01/seed_19/gb/result.json` | degree | 0.08 / 120 | 0.26 / 30 | none beyond historical provenance |
| `efficiency/citeseer/ptb_0.01/seed_19/kmeans/result.json` | degree | 0.08 / 120 | 0.26 / 30 | none beyond historical provenance |
| `efficiency/citeseer/ptb_0.01/seed_19/node_level/result.json` | degree | 0.08 / 120 | 0.26 / 30 | none beyond historical provenance |
| `efficiency/citeseer/ptb_0.05/seed_15/gb/result.json` | degree | 0.08 / 120 | 0.26 / 30 | none beyond historical provenance |
| `efficiency/citeseer/ptb_0.05/seed_15/kmeans/result.json` | degree | 0.08 / 120 | 0.26 / 30 | none beyond historical provenance |
| `efficiency/citeseer/ptb_0.05/seed_15/node_level/result.json` | degree | 0.08 / 120 | 0.26 / 30 | none beyond historical provenance |
| `efficiency/citeseer/ptb_0.05/seed_16/gb/result.json` | degree | 0.08 / 120 | 0.26 / 30 | none beyond historical provenance |
| `efficiency/citeseer/ptb_0.05/seed_16/kmeans/result.json` | degree | 0.08 / 120 | 0.26 / 30 | none beyond historical provenance |
| `efficiency/citeseer/ptb_0.05/seed_16/node_level/result.json` | degree | 0.08 / 120 | 0.26 / 30 | none beyond historical provenance |
| `efficiency/citeseer/ptb_0.05/seed_17/gb/result.json` | degree | 0.08 / 120 | 0.26 / 30 | none beyond historical provenance |
| `efficiency/citeseer/ptb_0.05/seed_17/kmeans/result.json` | degree | 0.08 / 120 | 0.26 / 30 | none beyond historical provenance |
| `efficiency/citeseer/ptb_0.05/seed_17/node_level/result.json` | degree | 0.08 / 120 | 0.26 / 30 | none beyond historical provenance |
| `efficiency/citeseer/ptb_0.05/seed_18/gb/result.json` | degree | 0.08 / 120 | 0.26 / 30 | none beyond historical provenance |
| `efficiency/citeseer/ptb_0.05/seed_18/kmeans/result.json` | degree | 0.08 / 120 | 0.26 / 30 | none beyond historical provenance |
| `efficiency/citeseer/ptb_0.05/seed_18/node_level/result.json` | degree | 0.08 / 120 | 0.26 / 30 | none beyond historical provenance |
| `efficiency/citeseer/ptb_0.05/seed_19/gb/result.json` | degree | 0.08 / 120 | 0.26 / 30 | none beyond historical provenance |
| `efficiency/citeseer/ptb_0.05/seed_19/kmeans/result.json` | degree | 0.08 / 120 | 0.26 / 30 | none beyond historical provenance |
| `efficiency/citeseer/ptb_0.05/seed_19/node_level/result.json` | degree | 0.08 / 120 | 0.26 / 30 | none beyond historical provenance |
| `efficiency/citeseer/ptb_0.1/seed_15/gb/result.json` | degree | 0.08 / 120 | 0.26 / 30 | none beyond historical provenance |
| `efficiency/citeseer/ptb_0.1/seed_15/kmeans/result.json` | degree | 0.08 / 120 | 0.26 / 30 | none beyond historical provenance |
| `efficiency/citeseer/ptb_0.1/seed_15/node_level/result.json` | degree | 0.08 / 120 | 0.26 / 30 | none beyond historical provenance |
| `efficiency/citeseer/ptb_0.1/seed_16/gb/result.json` | degree | 0.08 / 120 | 0.26 / 30 | none beyond historical provenance |
| `efficiency/citeseer/ptb_0.1/seed_16/kmeans/result.json` | degree | 0.08 / 120 | 0.26 / 30 | none beyond historical provenance |
| `efficiency/citeseer/ptb_0.1/seed_16/node_level/result.json` | degree | 0.08 / 120 | 0.26 / 30 | none beyond historical provenance |
| `efficiency/citeseer/ptb_0.1/seed_17/gb/result.json` | degree | 0.08 / 120 | 0.26 / 30 | none beyond historical provenance |
| `efficiency/citeseer/ptb_0.1/seed_17/kmeans/result.json` | degree | 0.08 / 120 | 0.26 / 30 | none beyond historical provenance |
| `efficiency/citeseer/ptb_0.1/seed_17/node_level/result.json` | degree | 0.08 / 120 | 0.26 / 30 | none beyond historical provenance |
| `efficiency/citeseer/ptb_0.1/seed_18/gb/result.json` | degree | 0.08 / 120 | 0.26 / 30 | none beyond historical provenance |
| `efficiency/citeseer/ptb_0.1/seed_18/kmeans/result.json` | degree | 0.08 / 120 | 0.26 / 30 | none beyond historical provenance |
| `efficiency/citeseer/ptb_0.1/seed_18/node_level/result.json` | degree | 0.08 / 120 | 0.26 / 30 | none beyond historical provenance |
| `efficiency/citeseer/ptb_0.1/seed_19/gb/result.json` | degree | 0.08 / 120 | 0.26 / 30 | none beyond historical provenance |
| `efficiency/citeseer/ptb_0.1/seed_19/kmeans/result.json` | degree | 0.08 / 120 | 0.26 / 30 | none beyond historical provenance |
| `efficiency/citeseer/ptb_0.1/seed_19/node_level/result.json` | degree | 0.08 / 120 | 0.26 / 30 | none beyond historical provenance |
| `efficiency/cora/ptb_0.01/seed_15/gb/result.json` | uniform | 0.08 / 180 | 0.28 / 60 | uniform_global_invariance |
| `efficiency/cora/ptb_0.01/seed_15/kmeans/result.json` | uniform | 0.08 / 180 | 0.28 / 60 | uniform_global_invariance |
| `efficiency/cora/ptb_0.01/seed_15/node_level/result.json` | uniform | 0.08 / 180 | 0.28 / 60 | uniform_global_invariance |
| `efficiency/cora/ptb_0.01/seed_16/gb/result.json` | uniform | 0.08 / 180 | 0.28 / 60 | uniform_global_invariance |
| `efficiency/cora/ptb_0.01/seed_16/kmeans/result.json` | uniform | 0.08 / 180 | 0.28 / 60 | uniform_global_invariance |
| `efficiency/cora/ptb_0.01/seed_16/node_level/result.json` | uniform | 0.08 / 180 | 0.28 / 60 | uniform_global_invariance |
| `efficiency/cora/ptb_0.01/seed_17/gb/result.json` | uniform | 0.08 / 180 | 0.28 / 60 | uniform_global_invariance |
| `efficiency/cora/ptb_0.01/seed_17/kmeans/result.json` | uniform | 0.08 / 180 | 0.28 / 60 | uniform_global_invariance |
| `efficiency/cora/ptb_0.01/seed_17/node_level/result.json` | uniform | 0.08 / 180 | 0.28 / 60 | uniform_global_invariance |
| `efficiency/cora/ptb_0.01/seed_18/gb/result.json` | uniform | 0.08 / 180 | 0.28 / 60 | uniform_global_invariance |
| `efficiency/cora/ptb_0.01/seed_18/kmeans/result.json` | uniform | 0.08 / 180 | 0.28 / 60 | uniform_global_invariance |
| `efficiency/cora/ptb_0.01/seed_18/node_level/result.json` | uniform | 0.08 / 180 | 0.28 / 60 | uniform_global_invariance |
| `efficiency/cora/ptb_0.01/seed_19/gb/result.json` | uniform | 0.08 / 180 | 0.28 / 60 | uniform_global_invariance |
| `efficiency/cora/ptb_0.01/seed_19/kmeans/result.json` | uniform | 0.08 / 180 | 0.28 / 60 | uniform_global_invariance |
| `efficiency/cora/ptb_0.01/seed_19/node_level/result.json` | uniform | 0.08 / 180 | 0.28 / 60 | uniform_global_invariance |
| `efficiency/cora/ptb_0.05/seed_15/gb/result.json` | uniform | 0.08 / 180 | 0.28 / 60 | uniform_global_invariance |
| `efficiency/cora/ptb_0.05/seed_15/kmeans/result.json` | uniform | 0.08 / 180 | 0.28 / 60 | uniform_global_invariance |
| `efficiency/cora/ptb_0.05/seed_15/node_level/result.json` | uniform | 0.08 / 180 | 0.28 / 60 | uniform_global_invariance |
| `efficiency/cora/ptb_0.05/seed_16/gb/result.json` | uniform | 0.08 / 180 | 0.28 / 60 | uniform_global_invariance |
