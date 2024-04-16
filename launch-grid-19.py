import subprocess
import shlex
import time
import os

commands = [
    # MIG 1 (0-7)
    ['MIG-c3ce33ce-ced8-5961-bb87-2b40eb100277', "python run_mctgraph.py 0 29500 -l -d 0.0 --exp_id='ct28_agent1_left'"],

    # MIG 2 (0-8)
    ['MIG-280489c4-1d98-5b07-b4f6-2fc85fc874fa', "python run_mctgraph.py 1 29501 -l -d 0.0 --exp_id='ct28_agent2_left'"],

    # MIG 3 (0-9)
    ['MIG-c432df19-0894-5232-ac1c-9a3440fc267e', "python run_mctgraph.py 2 29502 -l -d 0.0 --exp_id='ct28_agent3_left'"],

    # MIG 4 (0-10)
    ['MIG-e8f61a95-352a-56cc-b95d-0c35fc14e8bf', "python run_mctgraph.py 3 29503 -l -d 0.0 --exp_id='ct28_agent4_left'"],

    # MIG 5 (0-11)
    ['MIG-35ecef79-db2e-590b-9e8c-2c07c787008e', "python run_mctgraph.py 4 29504 -l -d 0.0 --exp_id='ct28_agent5_left'"],

    # MIG 6 (0-12)
    ['MIG-76cd8dd7-7703-5581-8ac5-a7ee81a402a0', "python run_mctgraph.py 5 29505 -l -d 0.0 --exp_id='ct28_agent6_left'"],

    # MIG 7 (0-13)
    ['MIG-b35e1a68-f7a4-5ef9-b34a-1abf6d1f8c2e', "python run_mctgraph.py 6 29506 -l -d 0.0 --exp_id='ct28_agent7_left'"],

    # MIG 8 (1-7)
    ['MIG-2d5b6364-fc42-587b-97c6-ee316a82e2f3', "python run_mctgraph.py 7 29507 -l -d 0.0 --exp_id='ct28_agent8_right'"],

    # MIG 9 (1-8)
    ['MIG-4590f80d-be70-58e4-af75-eeb950255d4a', "python run_mctgraph.py 8 29508 -l -d 0.0 --exp_id='ct28_agent9_right'"],

    # MIG 10 (1-9)
    ['MIG-e76a2a9b-9867-5f8a-b145-d857cd5ed8e2', "python run_mctgraph.py 9 29509 -l -d 0.0 --exp_id='ct28_agent10_right'"],

    # MIG 11 (1-10)
    ['MIG-2593b912-5975-58e9-bc3d-495311cee807', "python run_mctgraph.py 10 29510 -l -d 0.0 --exp_id='ct28_agent11_right'"],
    
    # MIG 12 (1-11)
    ['MIG-51069529-f343-59c6-bac7-a75648296e7b', "python run_mctgraph.py 1 29511 -l -d 0.0 --exp_id='ct28_agent12_right'"],

    # MIG 13 (1-12)
    ['MIG-187573d8-7df7-5e5f-87b2-c8b8f73c54e7', "python run_mctgraph.py 12 29512 -l -d 0.0 --exp_id='ct28_agent13_right'"],

    # MIG 14 (1-13)
    ['MIG-3045e3dd-28b6-5ee8-96b5-60a085c9fcf1', "python run_mctgraph.py 26 29513 -l -e --exp_id='ct28_md_eval_full' --reference='reference_eval.csv'"]
]

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