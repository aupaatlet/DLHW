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
Use `pip show mplib` to find where `mplib` installed, then use `vim <path>` to modify. Press `i` to insert. Finally, press `Esc` and type `:wq` to finish modification.
用命令`pip show mplib`找`mplib`的安装路径（假设是`<path>`，应该为`.../mplib/planner.py`）并复制，然后用命令`vim <path>`查看该文件。按`i`进入编辑模式，找到71行并把`convex=True`注释掉，找到848行并删除`or collide`。最后，点击`Esc`并输入`:wq`。

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
