# L2D2-C
Lifelong Learning Distributed Decentralized Collective (L2D2-C) is a shared experience lifelong learning system.
A distributed lifelong learning framework where each agent is a lifelong learner capable of sharing and recieving knowledge about previously learned tasks from other agents.
Our publication can be found at [https://proceedings.mlr.press/v232/nath23a/nath23a.pdf](https://proceedings.mlr.press/v232/nath23a/nath23a.pdf)

## Agent Description
- Each lifelong learner agent is a [PPO](https://arxiv.org/abs/1707.06347) controller combined with [Modulating Masks](https://openreview.net/forum?id=V7tahqGrOq) algorithm for knowledge preservation across multiple tasks. We leverage the masking methodology to share policies across agents.
- Basline agents is a regular [PPO](https://arxiv.org/abs/1707.06347) controller. No knowledge preservation algorithm is present, hence it catastrophically forgets.

## Evaluation Domain
Agent-based (reinforcement) learning. The L2D2-C code is built on top of existing [DeepRL](https://github.com/ShangtongZhang/DeepRL) repository. The repository is extended on two front
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
To run two agents on localhost simply run the following commands on two seperate terminals. NOTE: that the agent_id acts as an index value for the selection of a curriculum from the provided shell.json configuration file. It is also used to create directories for each agent's logs.
```
Terminal 1:
python run_shell_dist_mp.py <agent_id> <port> -l

Terminal 2:
python run_shell_dist_mp.py <agent_id> <port> -l
```

To run multiple agents on seperate devices, please update the addresses.csv file to contain the IPs and ports for ALL of your devices. For example:
```
xxx.xxx.0.1, 29500
xxx.xxx.0.2, 29501
```
To then run two agents on two different devices simply run the following commands:
```
Device 1:
python run_shell_dist_mp.py 0 29500

Device 2:
python run_shell_dist_mp.py 1 29501
```


The agent can also be run in evaluation mode to evaluate a collective of agents by using the flag -e, --e, or --eval:
```
python run_shell_dist_mp.py <agent_id> <port> -e
```


Additional parameters are also available in the system
```
--num_agents: Modify the default value of the initial world size (default starts at 1)
--shell_config_path: Modify the default path to the shell configuration JSON.
--exp_id: A unique ID/name for an experiment. Can be useful to seperate logging
--eval: Launches in evaluation mode
--localhost: Launches in localhost mode. Can be useful for debugging
--shuffle: Randomly shuffles the curriculum from the shell.json configuration. Can be useful for testing.
--comm_interval: An integer value to indicate the number of communications to perform per task. The default is 5.
--device: An integer value to indicate the device selection. By default it will select the GPU if available. Otherwise a value of 0 will indicate CPU.
--reference: The file path to the .csv file containing the address table of bootstrapping agents. These are the addresses the agent will use to connect to an existing network, or form a new one.
```

### Configuring environments/curriculum
Curriculums and environments can be modified from the shell.json file. This file contains the curriculum for each agent and has the following structure:
```
{
    "seed": [9157, 9158, ...],
    "agents": [
        {
            "task_ids": [0, 4, 6, 7, 1, 2, 4, 7, 9, 2, 4, ...]
            "max_steps": ...
        },
        ...
    ],
    "env": {
        "env_name": "ctgraph",
        "env_config_path": "./env_configs/meta_ctgraph.json"
    }
}
```
In the example above, tasks that correspond to the task_ids would be located in ./env_configs/meta_ctgraph.json.

Note: Minigrid (with 3 simple crossing tasks) is specified as the default environment/tasks in `shell.json`. To run CT-graph experiments, update `shell.json` as shown below, or alternatively use the shell2x2, shell4x4, shell8x8, shell16x16, and shell32x32 JSON files. Numbers indicate the maximum number of agents and the number of tasks the JSONs are configured for. You may wish to use a fewer number of agents with more tasks. Currently the maximum number of tasks configured for the CT-graph be default is 32 though this can be modified. Please get in touch with us if you would like to experiment with different CT-graph configurations.

## Metrics/Plots
To compute metrics and generate plots in multi_process ShELL settings (i.e. using `run_shell_dist.py` script to run ShELL experiments), read the documenation in this [link](README_plots.md)

## Maintainers
The repository is currently developed and maintained by researchers from Loughborough Univeristy, Vanderbilt University, UC Riverside, and UT Dallas

## BibTex
To cite this work, please use the information below.

```
@InProceedings{pmlr-v232-nath23a,
  title = 	 {Sharing Lifelong Reinforcement Learning Knowledge via Modulating Masks},
  author =       {Nath, Saptarshi and Peridis, Christos and Ben-Iwhiwhu, Eseoghene and Liu, Xinran and Dora, Shirin and Liu, Cong and Kolouri, Soheil and Soltoggio, Andrea},
  booktitle = 	 {Proceedings of The 2nd Conference on Lifelong Learning Agents},
  pages = 	 {936--960},
  year = 	 {2023},
  editor = 	 {Chandar, Sarath and Pascanu, Razvan and Sedghi, Hanie and Precup, Doina},
  volume = 	 {232},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {22--25 Aug},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v232/nath23a/nath23a.pdf},
  url = 	 {https://proceedings.mlr.press/v232/nath23a.html},
  abstract = 	 {Lifelong learning agents aim to learn multiple tasks sequentially over a lifetime. This involves the ability to exploit previous knowledge when learning new tasks and to avoid forgetting. Recently, modulating masks, a specific type of parameter isolation approach, have shown promise in both supervised and reinforcement learning. While lifelong learning algorithms have been investigated mainly within a single-agent approach, a question remains on how multiple agents can share lifelong learning knowledge with each other. We show that the parameter isolation mechanism used by modulating masks is particularly suitable for exchanging knowledge among agents in a distributed and decentralized system of lifelong learners. The key idea is that isolating specific task knowledge to specific masks allows agents to transfer only specific knowledge on-demand, resulting in a robust and effective collective of agents.  We assume fully distributed and asynchronous scenarios with dynamic agent numbers and connectivity. An on-demand communication protocol ensures agents query their peers for specific masks to be transferred and integrated into their policies when facing each task. Experiments indicate that on-demand mask communication is an effective way to implement distributed and decentralized lifelong reinforcement learning, and provides a lifelong learning benefit with respect to distributed RL baselines such as DD-PPO, IMPALA, and PPO+EWC. The system is particularly robust to connection drops and demonstrates rapid learning due to knowledge exchange.}
}
```

## Bug Reporting
If you encounter any bugs using the code, please raise an issue in the repository on Github.

## Acknowledgement
This material is based upon work supported by the United States Air Force Research Laboratory (AFRL) and Defense Advanced Research Projects Agency (DARPA) under Contract No. HR00112190132.

Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the United States Air Force Research Laboratory (AFRL) and Defense Advanced Research Projects Agency (DARPA).
