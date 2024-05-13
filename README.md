# C3L 
C3L (Continual Composition Collective Learning) is a [ShELL (Shared Experience Lifelong Learning)](https://rdcu.be/dB9zt) system that uses functional composition of knowledge from various sources within the collective, to promote knowledge reuse, accelerate learning and improve adapatability.
ShELL encapsulates an emerging class of distributed lifelong learning systems where each agent is a lifelong learner capable of sharing and receiving knowledge about tasks from other agents.

## Agent Description
Each lifelong learner agent is a [PPO](https://arxiv.org/abs/1707.06347) controller combined with [Modulating Masks](https://arxiv.org/abs/2212.11110) algorithm for knowledge preservation across tasks.
Basline agents is a regular [PPO](https://arxiv.org/abs/1707.06347) controller. No knowledge preservation algorithm is present, hence it catastrophically forgets.

## Evaluation Domain
Agent-based (reinforcement) learning. The DMIU code is built on top of existing [DeepRL](https://github.com/ShangtongZhang/DeepRL) repository. The repository is extended on three front
1. Extension of RL agents to lifelong learning framework, methods and experiments.
2. Extension of 1. to shared experience lifelong learning framework.
3. Extension of 2. for functional composition via Wasserstein task embedding similarity metrics.

Sample environments: [Minigrid](https://github.com/Farama-Foundation/gym-minigrid), [CT-graph](https://github.com/soltoggio/CT-graph), [Procgen](https://github.com/openai/procgen)

## Requirements
- Same as the requirements of the [DeepRL](https://github.com/ShangtongZhang/DeepRL) repository.
- [Minigrid](https://github.com/Farama-Foundation/gym-minigrid) if running minigrid experiments.
- [CT-graph](https://github.com/soltoggio/CT-graph) if running CT-graph experiments.
- [Procgen](https://github.com/openai/procgen) if running Procgen experiments.

- A full list of packages used in this codebase can be found in the ./ymls folder. These ymls can be used to create suitable Conda environments.

## Usage
To run a single C3L agent on [Minigrid](https://github.com/Farama-Foundation/gym-minigrid).
```
python run_minigrid.py <curriculum index> <listening port> -p <experiment folder name>
```
- The curriculum index tells the agent which curriculum of tasks to select for learning in the experiment.
- The listening port defines which port the server will listen on for incoming communication.
- The -p argument is optional and will default to the environment name.


[CT-graph](https://github.com/soltoggio/CT-graph) and [Procgen](https://github.com/openai/procgen) experiments can be run using run_mctgraph.py and run_procgen.py


To run a multi-agent experiment.
```
python launcher.py --env minigrid --exp <experiment folder name>
```
- The --env argument defines which setup to use. Each experiment has its own setup.
- The --exp argument defines the name of the folder in which the experiment data will be contained.
- The launcher.py file is setup to use CUDA_VISIBLE_DEVICES to define the GPU used by the agent. Our experiments have been run on Nvidia A100s using MiG configurations.


To run multiple agents on localhost, first check that the references.csv file contains the IPs and ports for your agents. By default this should look similar to this:

```
127.0.0.1, 29500
127.0.0.1, 29501
127.0.0.1, 29502
127.0.0.1, 29503
127.0.0.1, 29504
127.0.0.1, 29505
```
To run two agents on localhost simply run the following commands on two seperate terminals. NOTE: that the curriculum index acts as an index value for the selection of a curriculum from the provided shell.json configuration file. It is also used to create directories for each agent's logs.
```
Terminal 1:
python run_minigrid.py <curriculum index> <port value> -l

Terminal 2:
python run_minigrid.py <curriculum index> <port value> -l
```

To run multiple agents on seperate devices, please update the addresses.csv file to contain the IPs and ports for ALL of your devices. For example:
```
xxx.xxx.x.x, 29500
xxx.xxx.x.x, 29501
```
To then run two agents on two different devices simply run the following commands:
```
Device 1:
python run_minigrid.py 0 29500

Device 2:
python run_minigrid.py 1 29501
```

Additional parameters are also available in the system
```
--num_agents: Modify the default value of the initial world size (default starts at 1)
--shell_config_path: Modify the default path to the shell configuration JSON.
--exp_id: A unique ID/name for an experiment. Can be useful to seperate logging
--eval: Launches in evaluation mode
--localhost: Launches in localhost mode. Can be useful for debugging
--shuffle: Randomly shuffles the curriculum from the shell.json configuration. Can be useful for testing.
--comm_interval: An integer value to indicate the number of communications to perform per task.
--device: An integer value to indicate the device selection. By default it will select the GPU if available. Otherwise a value of 0 will indicate CPU.
--reference: The file path to the .csv file containing the address table of bootstrapping agents. These are the addresses the agent will use to connect to an existing network, or form a new one.
```

### Configuring environments/curriculum
Curriculums and environments can be modified from the shell.json file. This file contains the curriculum for each agent by

Note: Minigrid (with 3 simple crossing tasks) is specified as the default environment/tasks in `shell.json`. To run CT-graph experiments, update `shell.json` as shown below, or alternatively use the shell2x2, shell4x4, shell8x8, shell16x16, and shell32x32 JSON files. Numbers indicate the maximum number of agents and the number of tasks the JSONs are configured for. You may wish to use a smaller number of agents with more tasks. Currently, the maximum number of tasks configured for the CT-graph by default is 32 though this can be modified. Please get in touch with us if you would like to experiment with different CT-graph configurations.
```
...
"env": {
    "env_name": "ctgraph",
    "env_config_path": "./env_configs/meta_ctgraph.json"
}
...
```

## Maintainers
The repository is currently developed and maintained by researchers from Loughborough University, Vanderbilt University, UC Riverside, and UT Dallas

## Bug Reporting
If you encounter any bugs using the code, please raise an issue in the repository on GitHub.

## Acknowledgement
This material is based upon work supported by the United States Air Force Research Laboratory (AFRL) and Defense Advanced Research Projects Agency (DARPA) under Contract No. HR00112190132.

Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the United States Air Force Research Laboratory (AFRL) and Defense Advanced Research Projects Agency (DARPA).
