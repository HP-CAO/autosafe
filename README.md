# AutoSafe

![Tensorflow](https://img.shields.io/badge/Tensorflow-2.16-orange?logo=tensorflow)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Linux](https://img.shields.io/badge/Linux-22.04-yellow?logo=linux)
## Overview

A simple safe architecture for neural policy.


[//]: # (  <p align="center">)

[//]: # (  <p align="center">)

[//]: # (  <img src="frame.png" alt="Logo" width="90%"><br />)

[//]: # (  </p>)


## Set up
### Environment
This project is using the following settings:
- Ubuntu: 24.04 
- python: 3.11

### Create a virtual environment

```shell
conda create -n autosafe python=3.11
```

### Install packages in python

```shell
pip install -r requirement.txt
```

## Run
### Generate configuration files
Generate configuration file. You need to define your default job configurations by importing configurations from your job script.  
```shell 
python job/generate_config.py
```

### Start a job
Start a job with a specified configuration file and overriding parameter
```shell
python job/experiments/cartpole.py JobParams.job_name=base_cartpole_fix JobParams.experiment_name=base
```

We are using hydra to manager the configurations, you can run any job with --help to get more information 
```shell
python job/experiments/cartpole.py --help
```

### Monitor the training progress using tensorboard
```shell
tensorboard --logdir ./outputs/{FOLDER_NAME}
```


### Todos

- [x] Add BiGlucose and Glucose envs and test
- [x] Add Quadruped 
- [ ] Modify Quadruped to accommodate AutoSafe
- [x] Add PPO algorithm baseline
- [ ] Implement PP0 + AutoSafe

- [ ] Implement CPO (model-free)
- [ ] Implement Lyapunov-based Safe RL (model-free)
- 
- [ ] Implement AR-RL (model-based)
- [ ] Implement Simplex (model-based)
- [ ] Implement Residual (model-based)

