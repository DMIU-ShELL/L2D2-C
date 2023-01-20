import subprocess
import shlex
import time
import os

commands = []

'''# SHELL LEARNERS + EVALUATION AGENT
# Configuration
shell_path = 'shell_16x16.json'
num_agents = 16
localhost = False

addresses, ports = [], []
reference_file = open('reference.csv r')
lines = reference_file.readlines()

for line in lines:
    line = line.strip('\n').split( )
    addresses.append(line[0])
    ports.append(int(line[1]))

# Append the external agent commands
for i, port in enumerate(ports):
    # Learning agent
    if i < num_agents:
        commands.append(f'python run_shell_dist_mp.py {i} {port} -r reference.csv')

    # Evaluation agent
    else:
        commands.append(f'python run_shell_dist_mp.py {i} {port} -r reference.csv -e')'''


# SINGLE AGENT EXPERIMENT FOR COMPARISON
# Configuration for the accompanying single agent
'''
addresses_s, ports_s = [], []
reference_file = open('reference_single.csv r')
lines = reference_file.readlines()
for line in lines:
    line = line.strip('\n').split( )
    addresses_s.append(line[0])
    ports_s.append(int(line[1]))

for port_s in ports_s:
    if port_s in ports:
        print(f'ERROR: port {port_s} in reference_single.csv is also used in reference.csv. Please use a different port for your single agent experiment')

commands.append(f'python run_shell_dist_mp.py 0 {ports_s[0]} -r reference_single.csv')
commands.append(f'python run_shell_dist_mp.py 0 {ports_s[1]} -r reference_single.csv -e')
'''

commands = [
    # Shell Evaluator
    ['MIG-4590f80d-be70-58e4-af75-eeb950255d4a', 'python run_shell_dist_mp.py 0 29516 -e'],

    # Shell system experiment x18 agents
    ['MIG-c3ce33ce-ced8-5961-bb87-2b40eb100277', 'python run_shell_dist_mp.py 0 29500'],
    ['MIG-c3ce33ce-ced8-5961-bb87-2b40eb100277', 'python run_shell_dist_mp.py 1 29501'],

    ['MIG-280489c4-1d98-5b07-b4f6-2fc85fc874fa', 'python run_shell_dist_mp.py 2 29502'],
    ['MIG-280489c4-1d98-5b07-b4f6-2fc85fc874fa', 'python run_shell_dist_mp.py 3 29503'],

    ['MIG-c432df19-0894-5232-ac1c-9a3440fc267e', 'python run_shell_dist_mp.py 4 29504'],
    ['MIG-c432df19-0894-5232-ac1c-9a3440fc267e', 'python run_shell_dist_mp.py 5 29505'],

    ['MIG-e8f61a95-352a-56cc-b95d-0c35fc14e8bf', 'python run_shell_dist_mp.py 6 29506'],
    ['MIG-e8f61a95-352a-56cc-b95d-0c35fc14e8bf', 'python run_shell_dist_mp.py 7 29507'],

    ['MIG-35ecef79-db2e-590b-9e8c-2c07c787008e', 'python run_shell_dist_mp.py 8 29508'],
    ['MIG-35ecef79-db2e-590b-9e8c-2c07c787008e', 'python run_shell_dist_mp.py 9 29509'],

    ['MIG-76cd8dd7-7703-5581-8ac5-a7ee81a402a0', 'python run_shell_dist_mp.py 10 29510'],
    ['MIG-76cd8dd7-7703-5581-8ac5-a7ee81a402a0', 'python run_shell_dist_mp.py 11 29511'],

    ['MIG-b35e1a68-f7a4-5ef9-b34a-1abf6d1f8c2e', 'python run_shell_dist_mp.py 12 29512'],
    ['MIG-b35e1a68-f7a4-5ef9-b34a-1abf6d1f8c2e', 'python run_shell_dist_mp.py 13 29513'],

    ['MIG-2d5b6364-fc42-587b-97c6-ee316a82e2f3', 'python run_shell_dist_mp.py 14 29514'],
    ['MIG-2d5b6364-fc42-587b-97c6-ee316a82e2f3', 'python run_shell_dist_mp.py 15 29515']
]

env = dict(os.environ)
# Run the commands in seperate terminals
processes = []
for command in commands:
    print(command)
    env['CUDA_VISIBLE_DEVICES'] = command[0]
    process = subprocess.Popen(shlex.split(command[1]), env=env)
    processes.append(process)
    #time.sleep(2)

for process in processes:
    stdout, stderr = process.communicate()