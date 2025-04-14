# RoboTwin Data Generation Guide
## Step 1: Install Necessary Dependencies
* Follow the guide `Installation.md`

## Step 2: Modified task configure
* See `RoboTwin/CONFIG_TUTORIAL.md` to check the meaning of the parameters
* Open the file in `RoboTwin/task_config/<task_name>.yml`
* Set `save_freq=25` and `rgb=false`

## Step 3: Command
* Run `bash run_task.sh <task_name> <gpu_id>`, it will generate `.pkl` data
* The output should look like
```
...
simulate data episode 0 success! (seed = 0)
...
```

