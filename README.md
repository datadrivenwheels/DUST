# DUST
**Dual Swin Transformer for Video-Time-Series Fusion**

This project involves the use of a Dual Swin Transformer model to effectively fuse video and time-series data. Below are the steps for preparation, data generation, and training.

## Environment Setup

Follow these steps to set up the environment for the DUST project:

### Creating a Conda Environment

1. Create a new conda environment with Python 3.8:
   
   ```bash
   conda create -n dust python=3.8
   ```
2. Activate the newly created environment:
   
   ```bash
   conda activate dust
   ```
3. Install PyTorch, torchvision, torchaudio, and CUDA support using conda:

   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   ```
4. Install the necessary Python packages using pip:

   ```bash
   pip install timm
   pip install einops
   pip install opencv-python
   pip install pandas
   ```
   
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
         - `--lr`: Specifies the learning rate (default: 0.001).
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

   - step_fusion_3d_get_feature.py: Script Configuration

   - The `step_fusion_3d_get_feature.py` script is designed for extracting features from videos as part of the DUST model's training process. It utilizes several arguments for configuration:

      - Basic Settings
        
         - `--batch_size`: Sets the input batch size for training (default: 80).
         - `--seed`: Specifies a random seed for reproducibility (default: 93728645).

      - Model Configuration
         
         - `--path_3d_model`: Path to the pre-trained swin 3d model (default: './saved_models/step_3d/final_3d.pth').
         - `--num_frames`: Number of frames or time points (default: 77).
         - `--height`: Height of each frame (default: 224).
         - `--width`: Width of each frame (default: 224).

      - Input Paths
         
         - `--path_video_train`: Location of the video training data (default: './data/video_train.pkl').
         - `--path_video_val`: Location of the video validation data (default: './data/video_val.pkl').
         - `--path_label_train`: Path to training label data (default: './data/label_train.pkl').
         - `--path_label_val`: Path to validation label data (default: './data/label_val.pkl').

      - Output Paths
         
         - `--outpath`: Directory to save extracted video features (default: './video_features/').
         - `--logpath`: Directory for saving logs (default: './logs/').

   - These configurations enable the extraction of features from video data.


4. **Stage 4: Update Time-Series Model**
   - Finally, update the time-series model by running:
     ```
     python step_fusion.py
     ```
   - step_fusion.py: Script Configuration

   - The `step_fusion.py` script is an essential component for the final stage of the DUST model's training, focusing on the fusion of video and kinematic data. It is configured through the following arguments:

      - Training Parameters
         
         - `--batch_size`: Sets the batch size for training (default: 2000).
         - `--num_epochs`: Determines the number of training epochs (default: 200).
         - `--lr`: Specifies the learning rate (default: 0.0003).
         - `--seed`: Sets a random seed for reproducibility (default: 93728645).
         - `--print_every`: Defines the interval of epochs for validation and model saving (default: 1).

      - Model Configuration
         
         - `--path_1d_model`: Path to the pre-trained swin 1d model (default: './saved_models/step_1d/final_1d.pth').
         - `--num_frames`: Number of frames or time points (default: 51).
         - `--num_channels`: Number of channels (default: 3).

      - Input Paths
         
         - `--path_kine_train`: Location of the kinematic training data (default: './data/kine_train.pkl').
         - `--path_kine_val`: Location of the kinematic validation data (default: './data/kine_val.pkl').
         - `--path_vidfeat_train`: Path to the training video features (default: './video_features/train_3d_features.pkl').
         - `--path_vidfeat_val`: Path to the validation video features (default: './video_features/val_3d_features.pkl').
         - `--path_label_train`: Path to training label data (default: './data/label_train.pkl').
         - `--path_label_val`: Path to validation label data (default: './data/label_val.pkl').

      - Output Paths
         
         - `--outpath`: Directory to save the final fused models (default: './saved_models/step_fusion/').
         - `--logpath`: Directory for saving logs (default: './logs/').

This configuration caters to the intricate process of fusing video and kinematic data, representing the culmination of the DUST model's training protocol.

## Acknowledgments

This project has been greatly influenced and derives key components from the following repositories:

1. **Video Swin Transformer in PyTorch**: Our work incorporates elements from the Video Swin Transformer implementation available at [haofanwang/video-swin-transformer-pytorch](https://github.com/haofanwang/video-swin-transformer-pytorch). We extend our gratitude to Haofan Wang and contributors for their pioneering work in the field of video processing with transformers.

2. **Swin Transformer 1D**: Elements of our project are inspired by the Swin Transformer 1D, accessible at [meraks/Swin-Transformer-1D](https://github.com/meraks/Swin-Transformer-1D). We acknowledge the efforts of Merak and their contributors in adapting Swin Transformer architectures for 1D data processing, which has significantly informed our approach.

We thank these contributors for their open-source work, which has been instrumental in shaping our project.

