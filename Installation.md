# Step 1: Checking Dependency
* Python 3.8 or 3.10
`python--version`
* Operating systems: Linux: Ubuntu 18.04+, Centos 7+
`cat /etc/os-release`
* Hardware： NVIDIA GPU with recommended CUDA version 12.1
`nvcc --version`

# Step 2: Install Vulkan
`apt install libvulkan1 mesa-vulkan-drivers vulkan-tools`

# Step 3: Build Basic Environment
* Prepare a conda environment
`conda create -n RoboTwin python=3.8
conda activate RoboTwin`
* Install packages
`pip install torch==2.4.1 torchvision sapien==3.0.0b1 scipy==1.10.1 mplib==0.1.1 gymnasium==0.29.1 trimesh==4.4.3 open3d==0.18.0 imageio==2.34.2 pydantic zarr openai huggingface_hub==0.25.0`
* Clone RoboTwin GitHub
`git clone https://github.com/TianxingChen/RoboTwin`
* Install pytorch3d
`cd ../..
cd RoboTwin/third_party/pytorch3d_simplified
pip install -e .`

# Step 4: Download Assert
`cd ../..
cd RoboTwin/script
python download_asset.py
unzip aloha_urdf.zip
unzip main_models.zip`

# Step 5: Baselines (Optional)
* Install DP
`cd ../..
cd RoboTwin/policy/Diffusion-Policy
pip install -e .
`
* Install DP3
    * Install dp3
    `cd ../..
    cd RoboTwin/policy/3D-Diffusion-Policy/3D-Diffusion-Policy
    pip install -e .`
    * Install some necessary package
    `cd ../..
    pip install zarr==2.12.0 wandb ipdb gpustat dm_control omegaconf hydra-core==1.2.0 dill==0.3.5.1 einops==0.4.1 diffusers==0.11.1 numba==0.56.4 moviepy imageio av matplotlib termcolor
    cd ../..`
    
