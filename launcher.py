import subprocess
import shlex
import time
import os

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

# FOCCAL
commands = [
    # MIG 14 (1-13)
    ['MIG-3045e3dd-28b6-5ee8-96b5-60a085c9fcf1', 'python run_shell_dist_mp.py 0 29513 -e'],
    
    # MIG 1 (0-7)
    ['MIG-c3ce33ce-ced8-5961-bb87-2b40eb100277', 'python run_shell_dist_mp.py 1 29500 -d 0.0 -a 0.0'],

    # MIG 2 (0-8)
    ['MIG-280489c4-1d98-5b07-b4f6-2fc85fc874fa', 'python run_shell_dist_mp.py 2 29501 -d 0.0 -a 0.0'],

    # MIG 3 (0-9)
    ['MIG-c432df19-0894-5232-ac1c-9a3440fc267e', 'python run_shell_dist_mp.py 3 29502 -d 0.0 -a 0.0'],

    # MIG 4 (0-10)
    ['MIG-e8f61a95-352a-56cc-b95d-0c35fc14e8bf', 'python run_shell_dist_mp.py 4 29503 -d 0.0 -a 0.0'],

    # MIG 5 (0-11)
    ['MIG-35ecef79-db2e-590b-9e8c-2c07c787008e', 'python run_shell_dist_mp.py 5 29504 -d 0.0 -a 0.0'],

    # MIG 6 (0-12)
    ['MIG-76cd8dd7-7703-5581-8ac5-a7ee81a402a0', 'python run_shell_dist_mp.py 6 29505 -d 0.0 -a 0.0'],

    # MIG 7 (0-13)
    ['MIG-b35e1a68-f7a4-5ef9-b34a-1abf6d1f8c2e', 'python run_shell_dist_mp.py 7 29506 -d 0.0 -a 0.0'],

    # MIG 8 (1-7)
    ['MIG-2d5b6364-fc42-587b-97c6-ee316a82e2f3', 'python run_shell_dist_mp.py 8 29507 -d 0.0 -a 0.0'],

    # MIG 9 (1-8)
    ['MIG-4590f80d-be70-58e4-af75-eeb950255d4a', 'python run_shell_dist_mp.py 9 29508 -d 0.0 -a 0.0'],

    # MIG 10 (1-9)
    ['MIG-e76a2a9b-9867-5f8a-b145-d857cd5ed8e2', 'python run_shell_dist_mp.py 10 29509 -d 0.0 -a 0.0'],

    # MIG 11 (1-10)
    ['MIG-2593b912-5975-58e9-bc3d-495311cee807', 'python run_shell_dist_mp.py 11 29510 -d 0.0 -a 0.0'],

    # MIG 12 (1-11)
    ['MIG-51069529-f343-59c6-bac7-a75648296e7b', 'python run_shell_dist_mp.py 12 29511 -d 0.0 -a 0.0'],

    # MIG 13 (1-12)
    #['MIG-187573d8-7df7-5e5f-87b2-c8b8f73c54e7', 'python run_shell_dist_mp.py 12 29512 -d 0.0 -a 0.0'],

    
]

'''
# JAMMY
commands = [
    # GPU 0
    ['GPU-a36275d0-e3ba-2bc3-db53-23ddf54c4f96', 'python run_shell_dist_mp.py 70 29500'],
    ['GPU-a36275d0-e3ba-2bc3-db53-23ddf54c4f96', 'python run_shell_dist_mp.py 71 29501'],
    ['GPU-a36275d0-e3ba-2bc3-db53-23ddf54c4f96', 'python run_shell_dist_mp.py 72 29502'],
    ['GPU-a36275d0-e3ba-2bc3-db53-23ddf54c4f96', 'python run_shell_dist_mp.py 73 29503'],
    ['GPU-a36275d0-e3ba-2bc3-db53-23ddf54c4f96', 'python run_shell_dist_mp.py 74 29504'],
    ['GPU-a36275d0-e3ba-2bc3-db53-23ddf54c4f96', 'python run_shell_dist_mp.py 75 29505'],
    ['GPU-a36275d0-e3ba-2bc3-db53-23ddf54c4f96', 'python run_shell_dist_mp.py 76 29506'],
    ['GPU-a36275d0-e3ba-2bc3-db53-23ddf54c4f96', 'python run_shell_dist_mp.py 77 29507'],
    ['GPU-a36275d0-e3ba-2bc3-db53-23ddf54c4f96', 'python run_shell_dist_mp.py 78 29508'],
    ['GPU-a36275d0-e3ba-2bc3-db53-23ddf54c4f96', 'python run_shell_dist_mp.py 79 29509'],
    ['GPU-a36275d0-e3ba-2bc3-db53-23ddf54c4f96', 'python run_shell_dist_mp.py 80 29510'],
    ['GPU-a36275d0-e3ba-2bc3-db53-23ddf54c4f96', 'python run_shell_dist_mp.py 81 29511'],
    ['GPU-a36275d0-e3ba-2bc3-db53-23ddf54c4f96', 'python run_shell_dist_mp.py 82 29512'],
    ['GPU-a36275d0-e3ba-2bc3-db53-23ddf54c4f96', 'python run_shell_dist_mp.py 83 29513'],
    ['GPU-a36275d0-e3ba-2bc3-db53-23ddf54c4f96', 'python run_shell_dist_mp.py 84 29514'],
    ['GPU-a36275d0-e3ba-2bc3-db53-23ddf54c4f96', 'python run_shell_dist_mp.py 85 29515'],

    # GPU 1
    ['GPU-48574e02-3995-066a-0264-24854e47089f', 'python run_shell_dist_mp.py 86 29516'],
    ['GPU-48574e02-3995-066a-0264-24854e47089f', 'python run_shell_dist_mp.py 87 29517'],
    ['GPU-48574e02-3995-066a-0264-24854e47089f', 'python run_shell_dist_mp.py 88 29518'],
    ['GPU-48574e02-3995-066a-0264-24854e47089f', 'python run_shell_dist_mp.py 89 29519'],
    ['GPU-48574e02-3995-066a-0264-24854e47089f', 'python run_shell_dist_mp.py 90 29520'],
    ['GPU-48574e02-3995-066a-0264-24854e47089f', 'python run_shell_dist_mp.py 91 29521'],
    ['GPU-48574e02-3995-066a-0264-24854e47089f', 'python run_shell_dist_mp.py 92 29522'],
    ['GPU-48574e02-3995-066a-0264-24854e47089f', 'python run_shell_dist_mp.py 93 29523'],
    ['GPU-48574e02-3995-066a-0264-24854e47089f', 'python run_shell_dist_mp.py 94 29524'],
    ['GPU-48574e02-3995-066a-0264-24854e47089f', 'python run_shell_dist_mp.py 95 29525'],
    ['GPU-48574e02-3995-066a-0264-24854e47089f', 'python run_shell_dist_mp.py 96 29526'],
    ['GPU-48574e02-3995-066a-0264-24854e47089f', 'python run_shell_dist_mp.py 97 29527'],
    ['GPU-48574e02-3995-066a-0264-24854e47089f', 'python run_shell_dist_mp.py 98 29528'],
    ['GPU-48574e02-3995-066a-0264-24854e47089f', 'python run_shell_dist_mp.py 100 29529'],
    ['GPU-48574e02-3995-066a-0264-24854e47089f', 'python run_shell_dist_mp.py 101 29530'],
    ['GPU-48574e02-3995-066a-0264-24854e47089f', 'python run_shell_dist_mp.py 101 29531'],

    # GPU 2
    ['GPU-dcf44a28-dfb3-0d69-f9c8-2839bff43ce5', 'python run_shell_dist_mp.py 102 29532'],
    ['GPU-dcf44a28-dfb3-0d69-f9c8-2839bff43ce5', 'python run_shell_dist_mp.py 103 29533'],
    ['GPU-dcf44a28-dfb3-0d69-f9c8-2839bff43ce5', 'python run_shell_dist_mp.py 104 29534'],
    ['GPU-dcf44a28-dfb3-0d69-f9c8-2839bff43ce5', 'python run_shell_dist_mp.py 105 29535'],
    ['GPU-dcf44a28-dfb3-0d69-f9c8-2839bff43ce5', 'python run_shell_dist_mp.py 106 29536'],
    ['GPU-dcf44a28-dfb3-0d69-f9c8-2839bff43ce5', 'python run_shell_dist_mp.py 107 29537'],
    ['GPU-dcf44a28-dfb3-0d69-f9c8-2839bff43ce5', 'python run_shell_dist_mp.py 108 29538'],
    ['GPU-dcf44a28-dfb3-0d69-f9c8-2839bff43ce5', 'python run_shell_dist_mp.py 109 29539'],
    ['GPU-dcf44a28-dfb3-0d69-f9c8-2839bff43ce5', 'python run_shell_dist_mp.py 110 29540'],
    ['GPU-dcf44a28-dfb3-0d69-f9c8-2839bff43ce5', 'python run_shell_dist_mp.py 111 29541'],
    ['GPU-dcf44a28-dfb3-0d69-f9c8-2839bff43ce5', 'python run_shell_dist_mp.py 112 29542'],
    ['GPU-dcf44a28-dfb3-0d69-f9c8-2839bff43ce5', 'python run_shell_dist_mp.py 113 29543'],
    ['GPU-dcf44a28-dfb3-0d69-f9c8-2839bff43ce5', 'python run_shell_dist_mp.py 114 29544'],
    ['GPU-dcf44a28-dfb3-0d69-f9c8-2839bff43ce5', 'python run_shell_dist_mp.py 115 29545'],
    ['GPU-dcf44a28-dfb3-0d69-f9c8-2839bff43ce5', 'python run_shell_dist_mp.py 116 29546'],
    ['GPU-dcf44a28-dfb3-0d69-f9c8-2839bff43ce5', 'python run_shell_dist_mp.py 117 29547'],

    # GPU 3
    ['GPU-724a2221-8474-9db1-b503-f2708f36d317', 'python run_shell_dist_mp.py 118 29548'],
    ['GPU-724a2221-8474-9db1-b503-f2708f36d317', 'python run_shell_dist_mp.py 119 29549'],
    ['GPU-724a2221-8474-9db1-b503-f2708f36d317', 'python run_shell_dist_mp.py 120 29550'],
    ['GPU-724a2221-8474-9db1-b503-f2708f36d317', 'python run_shell_dist_mp.py 121 29551'],
    ['GPU-724a2221-8474-9db1-b503-f2708f36d317', 'python run_shell_dist_mp.py 122 29552'],
    ['GPU-724a2221-8474-9db1-b503-f2708f36d317', 'python run_shell_dist_mp.py 123 29553'],
    ['GPU-724a2221-8474-9db1-b503-f2708f36d317', 'python run_shell_dist_mp.py 124 29554'],
    ['GPU-724a2221-8474-9db1-b503-f2708f36d317', 'python run_shell_dist_mp.py 125 29555'],
    ['GPU-724a2221-8474-9db1-b503-f2708f36d317', 'python run_shell_dist_mp.py 126 29556'],
    ['GPU-724a2221-8474-9db1-b503-f2708f36d317', 'python run_shell_dist_mp.py 127 29557'],
    ['GPU-724a2221-8474-9db1-b503-f2708f36d317', 'python run_shell_dist_mp.py 128 29558'],
    ['GPU-724a2221-8474-9db1-b503-f2708f36d317', 'python run_shell_dist_mp.py 129 29559'],
    ['GPU-724a2221-8474-9db1-b503-f2708f36d317', 'python run_shell_dist_mp.py 130 29560'],
    ['GPU-724a2221-8474-9db1-b503-f2708f36d317', 'python run_shell_dist_mp.py 131 29561'],
    ['GPU-724a2221-8474-9db1-b503-f2708f36d317', 'python run_shell_dist_mp.py 132 8000'],
    ['GPU-724a2221-8474-9db1-b503-f2708f36d317', 'python run_shell_dist_mp.py 133 8888']
]

# BEAVER
commands = [
    # Shell Evaluator
    ['4', 'python run_shell_dist_mp.py 0 29500 -e'],
]
'''

env = dict(os.environ)
# Run the commands in seperate terminals
processes = []
for command in commands:
    print(command)
    env['CUDA_VISIBLE_DEVICES'] = command[0]
    process = subprocess.Popen(shlex.split(command[1]), env=env)
    processes.append(process)
    #time.sleep(5)

for process in processes:
    stdout, stderr = process.communicate()