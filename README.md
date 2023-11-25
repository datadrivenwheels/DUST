# DUST
Dual Swin Transformer for video-time-series fusion


## Preparation

`$ mkdir data logs saved_models video_features`
`$ mkdir saved_models/step_1d saved_models/step_3d saved_models/step_fusion`


## Generate Toy Data

`$ python gen_data.py`


## Training

- `$ python step_fusion_3d.py`: train video model (Stage 1)

- `$ python step_fusion_1d.py`: train time-series model (Stage 2)

- `$ python step_fusion_3d_get_feature.py`: extract feature from videos using model trained in Stage 1.

- `$ python step_fusion.py`: update time-series model (Stage 3)