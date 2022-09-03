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
- Same as the requirements of the [DeepRL](https://github.com/ShangtongZhang/DeepRL) repository.
- [Minigrid](https://github.com/Farama-Foundation/gym-minigrid) if running minigrid experiments.
- [CT-graph](https://github.com/soltoggio/CT-graph) if running CT-graph experiments.

## Usage
Note: Example commands below using [Minigrid](https://github.com/Farama-Foundation/gym-minigrid) environment.
To run a baseline agent, use the command below:
```
python run_ll.py baseline --env_name minigrid --env_config_path ./env_configs/minigrid_sc_3.json --max_steps 102400
```

To run a single lifelong learner agent, use the command below:
```
python run_ll.py supermask --env_name minigrid --env_config_path ./env_configs/minigrid_sc_3.json --max_steps 102400
```

To run a ShELL (multi lifelong learner agents) in a single process (sequential executions per shell train step), use the command below:
```
python run_shell.py --shell_config_path ./shell.json
```

To run a ShELL (multi lifelong learner agents) across multiple processes (concurrent executions per shell train step) in a single machine, use the command below:
```
python run_shell_dist.py --shell_config_path ./shell.json
```

Note: Minigrid (with 3 simple crossing tasks) is specified as the default environment/tasks in `shell.json`. To run CT-graph experiments, update `shell.json` as shown below.
```
...
"env": {
    "env_name": "ctgraph",
    "env_config_path": "./env_configs/meta_ctgraph.json"
}
...
```

## Metrics/Plots
To compute metrics and generate plots in multi_process ShELL settings (i.e. using `run_shell_dist.py` script to run ShELL experiments), read the documenation in this [link](README_plots.md)

## Maintainers
The repository is currently developed and maintained by researchers from Loughborough Univeristy, Vanderbilt University, UC Riverside, and UT Dallas

## Bug Reporting
If you encounter any bugs using the code, please raise an issue in the repository on Github.

## Acknowledgement
This material is based upon work supported by the United States Air Force Research Laboratory (AFRL) and Defense Advanced Research Projects Agency (DARPA) under Contract No. HR00112190132.

Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the United States Air Force Research Laboratory (AFRL) and Defense Advanced Research Projects Agency (DARPA).
