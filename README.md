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

   - step_fusion_3d.py: Script Configuration

   - The `step_fusion_3d.py` script is crucial for training the video model in the DUST framework. It includes several configuration options:

      - Training Parameters
     
         - `--batch_size`: Sets the batch size for training (default: 9).
         - `--num_epochs`: Determines the number of training epochs (default: 100).
         - `--lr`: Specifies the learning rate (default: 0.00001).
         - `--seed`: Sets a random seed for reproducibility (default: 93728645).
         - `--print_every`: Defines the interval of epochs for validation and model saving (default: 3).

      - Model Configuration
        
         - `--num_frames`: Number of frames or time points (default: 77).
         - `--height`: Height of each frame (default: 224).
         - `--width`: Width of each frame (default: 224).

      - Input Paths
    
         - `--path_video_train`: Path to the video training data (default: './data/video_train.pkl').
         - `--path_video_val`: Path to the video validation data (default: './data/video_val.pkl').
         - `--path_label_train`: Location of the training label data (default: './data/label_train.pkl').
         - `--path_label_val`: Location of the validation label data (default: './data/label_val.pkl').

      - Output Paths
         - `--outpath`: Directory to save the trained models (default: './saved_models/step_3d/').
         - `--logpath`: Directory for saving logs (default: './logs/').

   - These configurations allow for fine-tuning of the video model training process in the DUST project.


2. **Stage 2: Train Time-Series Model**
   - Next, train the time-series model using:
     ```
     python step_fusion_1d.py
     ```
   - step_fusion_1d.py: Script Configuration

   - The `step_fusion_1d.py` script, a key part of the DUST model's training, utilizes several arguments:

      - Training Parameters
         - `--batch_size`: Input batch size (default: 2000).
         - `--num_epochs`: Number of epochs (default: 200).
         - `--lr`: Learning rate (default: 0.0003).
         - `--seed`: Random seed (default: 93728645).
         - `--print_every`: Epoch interval for validation/model saving (default: 1).

      - Model Settings
         - `--num_frames`: Number of frames (default: 51).
         - `--num_channels`: Number of channels (default: 3).

      - Input/Output Paths
         - `--path_kine_train`: Kinematic training data path (default: './data/kine_train.pkl').
         - `--path_kine_val`: Kinematic validation data path (default: './data/kine_val.pkl').
         - `--path_label_train`: Training labels path (default: './data/label_train.pkl').
         - `--path_label_val`: Validation labels path (default: './data/label_val.pkl').
         - `--outpath`: Model saving directory (default: './saved_models/step_1d/').
         - `--logpath`: Log directory (default: './logs/').

   - These configurations provide detailed control for the time-series model training in the DUST framework.


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
