## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```


## Evaluation

To evaluate the model(s) in the paper, run this command:

```train
python meta.py --dataset <dataset>
```

The attack is generated with the existing GCN surrogate. To evaluate transfer
to independently retrained GIN and GraphSAGE target models, run:

```bash
python meta_gin.py --dataset <dataset>
python meta_gsage.py --dataset <dataset>
```

Each entry point reports clean, edge-only, feature-only, and combined attack
accuracy using the same data split and attack configuration as `meta.py`.

## Small datasets

The repository provides explicit experiment configurations for Cora-ML, ACM,
and PolBlogs:

```bash
./run_cora_ml.sh
./run_acm.sh
./run_polblogs.sh
```

Each script contains its own `ARGS` array so the complete dataset-specific
configuration is visible and editable in one place. The checked-in scripts run
`meta_gsage.py`; switch the final three lines when you want `meta.py` or
`meta_gin.py` instead.

The Python entry point downloads/loads the requested DeepRobust dataset in
`./Data/`. You can also prepare all three datasets explicitly:

```bash
python prepare_small_datasets.py --dataset cora_ml acm polblogs
```

Both `cora_ml` and the alias `cora-ml` are accepted by the Python entry points.

Cora-ML includes 2,879 `attr_names`; preparation creates
`./bow_cache/cora_ml.pkl` with exactly the same feature order. The DeepRobust
ACM package contains numeric features but no aligned feature names, while
PolBlogs has no node attributes and is loaded with identity features. Their
scripts therefore pass `--allow_fallback_vocabulary` explicitly: those two text
paths are feature-space ablations over `feature_0`, `feature_1`, ... placeholders,
not natural-language word attacks.

When `--allow_fallback_vocabulary` is explicit, an existing cache whose token
count does not match the dataset feature dimension is also replaced by a
feature-aligned placeholder vocabulary. Without that opt-in, the mismatch is
rejected.
