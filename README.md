# DUST
**Dual Swin Transformer for Video-Time-Series Fusion**

This project involves the use of a Dual Swin Transformer model to effectively fuse video and time-series data. Below are the steps for preparation, data generation, and training.

## Preparation
First, create the necessary directories for data storage, logs, and model saving. Run the following commands in your terminal:

`$ mkdir data logs saved_models video_features`

`$ mkdir saved_models/step_1d saved_models/step_3d saved_models/step_fusion`


## Generate Toy Data
To generate toy data for training and testing, run the following command:

`$ python gen_data.py`


## Training
The training process is divided into several stages:

1. **Stage 1: Train Video Model**
   - Train the video model by executing:
     ```
     python step_fusion_3d.py
     ```

2. **Stage 2: Train Time-Series Model**
   - Next, train the time-series model using:
     ```
     python step_fusion_1d.py
     ```

3. **Stage 3: Extract Video Features**
   - After training the video model, extract features from videos with:
     ```
     python step_fusion_3d_get_feature.py
     ```

4. **Stage 4: Update Time-Series Model**
   - Finally, update the time-series model by running:
     ```
     python step_fusion.py
     ```
