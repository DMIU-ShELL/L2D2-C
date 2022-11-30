# DMIU
Detect, Module, Integrate and Transfer unit (DMIU) is a Shared experience lifelong learning system.
A distributed lifelong learning framework where each agent is a lifelong learner capable of sharing and receiving knowledge about tasks from other agents.

## Agent Description
Each lifelong learner agent is a [PPO](https://arxiv.org/abs/1707.06347) controller combined with [Supermask Superposition](https://arxiv.org/abs/2006.14769) algorithm for knowledge preservation across tasks.
Basline agents is a regular [PPO](https://arxiv.org/abs/1707.06347) controller. No knowledge preservation algorithm is present, hence it catastrophically forgets.

## Evaluation Domain
Agent-based (reinforcement) learning. The DMIU code is built on top of existing [DeepRL](https://github.com/ShangtongZhang/DeepRL) repository. The repository is extended on two front
1. Extension of RL agents to lifelong learning framework, methods and experiments.
2. Extension of 1. to shared experience lifelong learning framework.

Sample environments: [Minigrid](https://github.com/Farama-Foundation/gym-minigrid), [CT-graph](https://github.com/soltoggio/CT-graph)

## Requirements
Please take a look at the envrionment.yml/environment_mac.yml to check the packages requirements for your conda environment. Important packages to look out for are the versions for Minigrid=1.1.0 and Gym=0.24.0.

- Same as the requirements of the [DeepRL](https://github.com/ShangtongZhang/DeepRL) repository.
- [Minigrid](https://github.com/Farama-Foundation/gym-minigrid) if running minigrid experiments.
- [CT-graph](https://github.com/soltoggio/CT-graph) if running CT-graph experiments.

## Usage
Note: Example commands below using [Minigrid](https://github.com/Farama-Foundation/gym-minigrid) environment.
To run a ShELL agent use the following command:
```
python run_shell_dist_mp.py <agent id> <number of agents in the network> --ip <ip address> --port <port value>
```
- The agent ID is used for both logging purposes and for selecting the correct curriculum from the shell.json file. Please set this correctly.
- The number of agents in the network indicates the maximum number of peers in the network. Simply put a value that matches to however many peers you would like to run in a network.
- IP and port are necessary for the listening server portion of the system. IP address is localhost by default but should be changed to the device IP if performing cross device experiments. Port must be specified and should ideally not be a privileged port. Currently the system has not been tested using privileged ports such as 443 (https).

There are some additional arguments available in the code, details of which can be found in the `run_shell_dist_mp.py` file.


To run multiple agents on localhost, first check that the addresses.csv file contains the IPs and ports for your agents. By default this should look like:

```
127.0.0.1, 29500
127.0.0.1, 29501
127.0.0.1, 29502
127.0.0.1, 29503
127.0.0.1, 29504
127.0.0.1, 29505
```
To run two agents on localhost simply run the following commands on two seperate terminals:
```
Terminal 1:
python run_shell_dist_mp.py 0 2 --port 29500

Terminal 2:
python run_shell_dist_mp.py 1 2 --port 29501
```

To run multiple agents on seperate devices, please update the addresses.csv file to contain the IPs and ports for ALL of your devices. For example:
```
xxx.xxx.0.1, 29500
xxx.xxx.0.2, 29501
```
To then run two agents on two different devices simply run the following commands:
```
Device 1:
python run_shell_dist_mp.py 0 2 --ip xxx.xxx.0.1 --port 29500

Device 2:
python run_shell_dist_mp.py 1 2 --ip xxx.xxx.0.2 --port 29501
```

In a future release we hope to remove the need for a look-up table by implementing an additional protocol for dynamically adding and removing agents!


### Configuring environments/curriculum
Curriculums and environments can be modified from the shell.json file. This file contains the curriculum for each agent by

Note: Minigrid (with 3 simple crossing tasks) is specified as the default environment/tasks in `shell.json`. To run CT-graph experiments, update `shell.json` as shown below, or alternatively use the shell2x2, shell4x4, shell8x8, shell16x16, and shell32x32 JSON files. Numbers indicate the maximum number of agents and the number of tasks the JSONs are configured for. You may wish to use a fewer number of agents with more tasks. Currently the maximum number of tasks configured for the CT-graph be default is 32 though this can be modified. Please get in touch with us if you would like to experiment with different CT-graph configurations.
```
...
"env": {
    "env_name": "ctgraph",
    "env_config_path": "./env_configs/meta_ctgraph.json"
}
...
```

## Metrics/Plots
To compute metrics and generate plots in multi_process ShELL settings (i.e. using `run_shell_dist.py` script to run ShELL experiments), read the documenation in this [link](README_plots.md) (potentially outdated information)

## Maintainers
The repository is currently developed and maintained by researchers from Loughborough Univeristy, Vanderbilt University, UC Riverside, and UT Dallas

## Bug Reporting
If you encounter any bugs using the code, please raise an issue in the repository on Github.

## Acknowledgement
This material is based upon work supported by the United States Air Force Research Laboratory (AFRL) and Defense Advanced Research Projects Agency (DARPA) under Contract No. HR00112190132.

Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the United States Air Force Research Laboratory (AFRL) and Defense Advanced Research Projects Agency (DARPA).
