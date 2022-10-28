import json
import random


if __name__ == '__main__':

    with open('shell_2t_30a.json', 'r') as f:
        shell_config = json.load(f)

        print(shell_config)
        shell_config['curriculum'] = shell_config['agents'][0]
        print()
        print(shell_config['curriculum'])
        print()
        print(shell_config['curriculum']['task_ids'])
        random.shuffle(shell_config['curriculum']['task_ids'])
        print(shell_config['curriculum']['task_ids'])
        shell_config['seed'] = shell_config['seed'][0]      # Chris
        del shell_config['agents'][0]