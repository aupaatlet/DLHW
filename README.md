# Readme

## Data Generation From RoboTwin
### Step 1: Checking Dependency
* Python 3.8 or 3.10
```
python--version
```
* Operating systems: Linux: Ubuntu 18.04+, Centos 7+
```
cat /etc/os-release
```
* Hardware： NVIDIA GPU with recommended CUDA version 12.1
```
nvcc --version
```

### Step 2: Install Vulkan (Must run every time)
```
apt install libvulkan1 mesa-vulkan-drivers vulkan-tools
apt install -y libx11-6 libgl1 libglx0 libegl1 libxext6 libxi6
```

### Step 3: Build Basic Environment
* Prepare a conda environment
```
conda create -p RoboTwin python=3.10 -y
conda activate /output/RoboTwin
```
* Install packages
```
pip install torch==2.4.1 torchvision sapien==3.0.0b1 scipy==1.10.1 mplib==0.1.1 gymnasium==0.29.1 trimesh==4.4.3 open3d==0.18.0 imageio==2.34.2 pydantic zarr openai huggingface_hub==0.25.0
```
* Clone RoboTwin GitHub
```
git clone https://github.com/TianxingChen/RoboTwin
```
* Install pytorch3d
```
cd ../.. 
cd RoboTwin/third_party/pytorch3d_simplified
pip install -e .
```

### Step 4: Download Assert
```
cd ../..
cd RoboTwin/script
python download_asset.py
unzip aloha_urdf.zip
unzip main_models.zip
```

### Step 5: Modify `mplib` Library Code
Use command `pip show mplib` to find the installed path of `mplib` (Assume `<path>`, it should be `.../mplib/planner.py`) then copy, use command `vim <path>` and press `i` to edit afterward. Finally, make changes for the line 71 (`convex=True` to `# convex=True` and line 848 remove `or collide`. Finish the changes by pressing `Esc` then type `:wq`.

* Remove `convex=True`

```
# mplib.planner (mplib/planner.py) line 71
# convex=True

self.robot = ArticulatedModel(
            urdf,
            srdf,
            [0, 0, -9.81],
            user_link_names,
            user_joint_names,
            convex=True,
            verbose=False,
        )
        
=> 

self.robot = ArticulatedModel(
            urdf,
            srdf,
            [0, 0, -9.81],
            user_link_names,
            user_joint_names,
            # convex=True,
            verbose=False,
        )
```

* Remove `or collide`

```
# mplib.planner (mplib/planner.py) line 848
# remove or collide

if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:
                return {"status": "screw plan failed"}
=>
if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:
                return {"status": "screw plan failed"}
```

### Step 6: Modify Task Configure
* Open the task configure files in `RoboTwin/task_config/<task name>.yml`.
* Modify the following parameters
```
...
pcd_down_sample_num: 0
save_freq: 25
...
```
* Run task using the command， it will generate `.pkl` files
```
cd RoboTwin
bash run_task.sh <task_name> <gpu_id>
```

### Step 7: Convert PKL into PNG
* The `.pkl` files are very large, we will convert them into `.png` files
* Open `pkl_to_png.ipynb`, modify your `.pkl` path and saving path, then run
* For each task, 100 observations (episodes) will be generated

## Finetune ip2p with our own dataset
### Step 1: Checking Dependency
*CUDA 12.1 and python 3.9(our code is based on GRMG)

### Step 2: Preparation
```bash
# clone this repository
git clone https://github.com/bytedance/GR-MG.git
cd GR_MG
# install dependencies for goal image generation model
bash ./goal_gen/install.sh
# install dependencies for multi-modal goal conditioned policy
bash ./policy/install.sh
```
Download the pretrained [InstructPix2Pix](https://huggingface.co/timbrooks/instruct-pix2pix) weights from Huggingface and save them in `resources/IP2P/`. 
Download the pretrained MAE encoder [mae_pretrain_vit_base.pth ](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) and save it in `resources/MAE/`.
Organize the dataset and save them in `resources/data`.

### Step 3: Training

```bash
# modify the variables in the script before you execute the following instruction
bash ./goal_gen/newtrain.sh  ./goal_gen/config/newtrain.json
```

### Step 4: Evaluate

```bash
# load the checpoint and get the predicted pics
python ceshi2.py
```
