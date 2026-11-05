# PrimitiveUDF V3 release

Only the exact V3 model is supported. Preserve checkpoint parameter names.
`train.py` is the training/validation entrypoint; `tools/reconstruct_mesh.py`
and `tools/evaluate_mesh.py` perform reconstruction and evaluation.
Keep the three dataset configs and pretrained manifest synchronized.
Run `python tools/check_pipeline_config.py` and `python -m unittest discover -s tests`.
Do not commit datasets, generated meshes, or experiment logs.
Checkpoints use Git LFS. Do not push without user approval.
