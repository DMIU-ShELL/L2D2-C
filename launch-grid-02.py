import subprocess
import shlex
import time
import os

commands = [
    # MIG 1 (0-7)
    ['MIG-c8cbc779-6499-5d1f-86e7-ad789e047331', "python run_shell_dist_mp.py 0 29500 -l -d 0.0 --exp_id='ct28_agent1_left'"],

    # MIG 2 (0-8)
    ['MIG-f6c4eed6-1edf-5515-8e0b-0a5521e81cbb', "python run_shell_dist_mp.py 1 29501 -l -d 0.0 --exp_id='ct28_agent2_left'"],

    # MIG 3 (0-9)
    ['MIG-ae777ac5-da68-5606-8581-1878288224dc', "python run_shell_dist_mp.py 2 29502 -l -d 0.0 --exp_id='ct28_agent3_left'"],

    # MIG 4 (0-10)
    ['MIG-b2a723b5-db7a-50d8-a267-70569dafe609', "python run_shell_dist_mp.py 3 29503 -l -d 0.0 --exp_id='ct28_agent4_left'"],

    # MIG 5 (0-11)
    ['MIG-73000f9d-03d4-5dab-ad23-34b546569bdd', "python run_shell_dist_mp.py 4 29504 -l -d 0.0 --exp_id='ct28_agent5_left'"],

    # MIG 6 (0-12)
    ['MIG-f86cf155-b80e-56da-92ed-03c78bf647c7', "python run_shell_dist_mp.py 5 29505 -l -d 0.0 --exp_id='ct28_agent6_left'"],

    # MIG 7 (0-13)
    ['MIG-b35e1a68-f7a4-5ef9-b34a-1abf6d1f8c2e', "python run_shell_dist_mp.py 6 29506 -l -d 0.0 --exp_id='ct28_agent7_left'"],

    # MIG 8 (1-7)
    ['MIG-f61216a8-2a31-500e-acf7-f0158fbf7ce3', "python run_shell_dist_mp.py 7 29507 -l -d 0.0 --exp_id='ct28_agent8_right'"],

    # MIG 9 (1-8)
    ['MIG-bc91396f-1c2b-5319-8862-6f16d089ce5e', "python run_shell_dist_mp.py 8 29508 -l -d 0.0 --exp_id='ct28_agent9_right'"],

    # MIG 10 (1-9)
    ['MIG-13d6b3aa-c302-5ac2-9183-5494d22547e6', "python run_shell_dist_mp.py 9 29509 -l -d 0.0 --exp_id='ct28_agent10_right'"],

    # MIG 11 (1-10)
    ['MIG-f2d8b14c-d00c-5b5f-be98-bd1bff3bf371', "python run_shell_dist_mp.py 10 29510 -l -d 0.0 --exp_id='ct28_agent11_right'"],
    
    # MIG 12 (1-11)
    ['MIG-62fabbf0-b8de-5040-b4db-93f62c477543', "python run_shell_dist_mp.py 1 29511 -l -d 0.0 --exp_id='ct28_agent12_right'"],

    # MIG 13 (1-12)
    ['MIG-fcda8ac4-e82d-5259-8495-c777d8c95d74', "python run_shell_dist_mp.py 12 29512 -l -d 0.0 --exp_id='ct28_agent13_right'"],

    # MIG 14 (1-13)
    ['MIG-f4aecf22-d8a0-50a9-b804-c65cb2d2ff7e', "python run_shell_dist_mp.py 26 29513 -l -e --exp_id='ct28_md_eval_full' --reference='reference_eval.csv'"]
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