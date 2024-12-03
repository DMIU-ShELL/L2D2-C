# from https://raw.githubusercontent.com/openai/baselines/master/baselines/bench/monitor.py

import gym
from gym.core import Wrapper
import time
from glob import glob
import csv
import os.path as osp
import os
import json
import numpy as np

class Monitor(Wrapper):
    EXT = "monitor.csv"
    f = None

    def __init__(self, env, filename, allow_early_resets=True, reset_keywords=(), info_keywords=()):
        super().__init__(env)
        self.tstart = time.time()

        # Set up episode-level logging
        if filename is None:
            self.f = None
            self.logger = None
            self.log_file = None
        else:
            if not filename.endswith(Monitor.EXT):
                if osp.isdir(filename):
                    filename = osp.join(filename, Monitor.EXT)
                else:
                    filename = filename + "." + Monitor.EXT
            self.f = open(filename, "wt")
            self.log_file = filename.replace(Monitor.EXT, "monitor_log.csv")
            self.f.write('#%s\n' % json.dumps({"t_start": self.tstart, 'env_id': env.spec and env.spec.id}))
            self.logger = csv.DictWriter(self.f, fieldnames=('r', 'l', 't') + reset_keywords + info_keywords)
            self.logger.writeheader()
            self.f.flush()

        # Set up step-level logging
        self.step_log_file = self.log_file  # Use the same base name for the step-level CSV
        self.step_log_writer = None
        if self.step_log_file:
            step_fieldnames = ["episode", "step", "action", "observation", "reward", "done", "truncated", "info"]
            self.step_log_writer = open(self.step_log_file, "wt", newline="")
            self.step_csv_writer = csv.DictWriter(self.step_log_writer, fieldnames=step_fieldnames)
            self.step_csv_writer.writeheader()
            self.step_log_writer.flush()

        self.reset_keywords = reset_keywords
        self.info_keywords = info_keywords
        self.allow_early_resets = allow_early_resets
        self.rewards = None
        self.needs_reset = True
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.total_steps = 0
        self.current_reset_info = {}  # Extra info about the current episode

    def reset(self, **kwargs):
        # Enforce early reset rules
        if not self.allow_early_resets and not self.needs_reset:
            raise RuntimeError(
                "Tried to reset an environment before done. If you want to allow early resets, "
                "wrap your env with Monitor(env, path, allow_early_resets=True)"
            )

        self.rewards = []
        self.needs_reset = False

        # Log reset keywords
        for k in self.reset_keywords:
            v = kwargs.get(k)
            if v is None:
                raise ValueError(f"Expected you to pass kwarg {k} into reset")
            self.current_reset_info[k] = v

        # Record reset data
        done = None
        reset_tuple = self.env.reset(**kwargs)
        if len(reset_tuple) == 2:
            state, done = reset_tuple
        else:
            state = reset_tuple
        
        if self.step_csv_writer:
            self.step_csv_writer.writerow({
                "episode": len(self.episode_rewards) + 1,
                "step": 0,
                "action": None,
                "observation": None,
                "reward": None,
                "done": done if done is not None else False,
                "truncated": False,
                "info": {}
            })
            self.step_log_writer.flush()

        return reset_tuple

    def step(self, action):
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")

        # Interact with the environment
        step_tuple = self.env.step(action)
        if len(step_tuple) == 4:
            ob, rew, done, info = step_tuple
            truncated = None
        elif len(step_tuple) == 5:
            ob, rew, done, truncated, info = step_tuple
        else:
            raise ValueError("Unexpected step tuple size; expected 4 or 5 elements.")

        # Log step data
        self.rewards.append(rew)
        self.total_steps += 1
        step_data = {
            "episode": len(self.episode_rewards) + 1,
            "step": len(self.rewards),
            "action": action,
            "observation": None,
            "reward": rew,
            "done": done,
            "truncated": truncated,
            "info": info
        }
        if self.step_csv_writer:
            self.step_csv_writer.writerow(step_data)
            self.step_log_writer.flush()

        # Handle episode end
        if done or truncated:
            self.needs_reset = True
            eprew = sum(self.rewards)
            eplen = len(self.rewards)
            epinfo = {"r": round(eprew, 6), "l": eplen, "t": round(time.time() - self.tstart, 6)}
            for k in self.info_keywords:
                epinfo[k] = info.get(k, None)

            self.episode_rewards.append(eprew)
            self.episode_lengths.append(eplen)
            self.episode_times.append(time.time() - self.tstart)
            epinfo.update(self.current_reset_info)

            if self.logger:
                self.logger.writerow(epinfo)
                self.f.flush()

            info['episode'] = epinfo

        return step_tuple

    def close(self):
        # Close log files and write final logs
        if self.f is not None:
            self.f.close()
        if self.step_log_writer is not None:
            self.step_log_writer.close()

    def get_total_steps(self):
        return self.total_steps

    def get_episode_rewards(self):
        return self.episode_rewards

    def get_episode_lengths(self):
        return self.episode_lengths

    def get_episode_times(self):
        return self.episode_times

class LoadMonitorResultsError(Exception):
    pass

def get_monitor_files(dir):
    return glob(osp.join(dir, "*" + Monitor.EXT))

def load_monitor_log(dir):
    import pandas
    monitor_files = (
        glob(osp.join(dir, "*monitor.json")) +
        glob(osp.join(dir, "*monitor.csv"))) # get both csv and (old) json files
    if not monitor_files:
        raise LoadMonitorResultsError("no monitor files of the form *%s found in %s" % (Monitor.EXT, dir))
    dfs = []
    headers = []
    for fname in monitor_files:
        with open(fname, 'rt') as fh:
            if fname.endswith('csv'):
                firstline = fh.readline()
                assert firstline[0] == '#'
                header = json.loads(firstline[1:])
                df = pandas.read_csv(fh, index_col=None)
                headers.append(header)
            elif fname.endswith('json'): # Deprecated json format
                episodes = []
                lines = fh.readlines()
                header = json.loads(lines[0])
                headers.append(header)
                for line in lines[1:]:
                    episode = json.loads(line)
                    episodes.append(episode)
                df = pandas.DataFrame(episodes)
            else:
                assert 0, 'unreachable'
            df['t'] += header['t_start']
        dfs.append(df)
    df = pandas.concat(dfs)
    df.sort_values('t', inplace=True)
    df.reset_index(inplace=True)
    df['t'] -= min(header['t_start'] for header in headers)
    # df.headers = headers # HACK to preserve backwards compatibility
    return df
