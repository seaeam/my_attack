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
