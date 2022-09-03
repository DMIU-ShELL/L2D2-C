# Generating Plots for Metrics (in Multi-Process ShELL Settings)

## Template Command
To compute metrics and generate plots specific to ShELL, you need results from ShELL experiment only. Use the command below (works with single or multiple seeds).
```
python plot_shell_metrics.py ./path-to-shell-seed1 ./path-to-shell-seed2  .....  ./path-to-shell-seedn
```

To compute metrics and generate the full set of plots (including comparison of ShELL against a single lifelong learning agent), you need results for both ShELL and single agent experiments. Use the command below (works with single or multiple seeds).
```
python plot_shell_metrics.py ./path-to-shell-seed1/ ./path-to-shell-seed2/  .....  ./path-to-shell-seedn/   --ll_paths ./path-to-ll-seed1/timestamp/ ./path-to-ll-seed2/timestamp/  .....  ./path-to-ll-seedn/timestamp/
```

Plots are generated and stored in a directory called `metrics_plot`. To change the name of the directory, use the `--exp_name` flag when running the plot metrics script/command above.

## Example Directory Structure and Command
Below is a sample log directory structure of a ShELL experiment (one seed run).

MetaCTgraph-shell-dist-upz-seed-8379/
├── agent_0/
│   └── 220902-132612/
│       ├── .....
│       ├── eval_metrics_agent_0.csv
│       ├── eval_metrics_agent_0.npy
│       ├── .....
│       └── 
└── agent_1/
    └── 220902-132617/
        ├── .....
        ├── eval_metrics_agent_1.csv
        ├── eval_metrics_agent_1.npy
        ├── .....
        └── 

Below is a sample log directory structure of a single agent experiment (one seed run).
MetaCTgraph-ppo-supermask-seed-8379/
└── 220902-135804/
    ├── .....
    ├── eval_metrics.csv
    ├── eval_metrics.npy
    ├── eval_stats/
    ├── task_stats/
    ├── .....
    └── 

and a sample command to generate and plot the metrics is given below.
```
plot_shell_metrics.py ./log/MetaCTgraph-shell-dist-upz-seed-9157/ --ll_paths ./log/MetaCTgraph-ppo-supermask-seed-8379/220902-135804/
```
