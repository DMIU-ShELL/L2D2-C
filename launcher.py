import subprocess
import shlex
import time
import os
import argparse

# FOCCAL (CT-Graph)
commands_ctgraph = [
    # IMG SEED 1 AGENTS
    # MIG 1 (0-7)
    ['MIG-c8cbc779-6499-5d1f-86e7-ad789e047331', "python run_mctgraph.py 0 29500 -l -d 0.0 --exp_id='ct28_agent0_left'"],

    # MIG 2 (0-8)
    ['MIG-f6c4eed6-1edf-5515-8e0b-0a5521e81cbb', "python run_mctgraph.py 1 29501 -l -d 0.0 --exp_id='ct28_agent1_left'"],

    # MIG 3 (0-9)
    ['MIG-ae777ac5-da68-5606-8581-1878288224dc', "python run_mctgraph.py 2 29502 -l -d 0.0 --exp_id='ct28_agent2_left'"],

    # MIG 4 (0-10)
    ['MIG-b2a723b5-db7a-50d8-a267-70569dafe609', "python run_mctgraph.py 3 29503 -l -d 0.0 --exp_id='ct28_agent3_left'"],

    # MIG 5 (0-11)
    ['MIG-73000f9d-03d4-5dab-ad23-34b546569bdd', "python run_mctgraph.py 4 29504 -l -d 0.0 --exp_id='ct28_agent4_left'"],

    # MIG 6 (0-12)
    ['MIG-f86cf155-b80e-56da-92ed-03c78bf647c7', "python run_mctgraph.py 5 29505 -l -d 0.0 --exp_id='ct28_agent5_left'"],

    # MIG 7 (0-13)
    ['MIG-bfb3fc5a-8b5e-59c1-87e8-3468c995103d', "python run_mctgraph.py 6 29506 -l -d 0.0 --exp_id='ct28_agent6_left'"]


    # IMG SEED 2 AGENTS
    # MIG 8 (1-7)
    #['MIG-f61216a8-2a31-500e-acf7-f0158fbf7ce3', "python run_mctgraph.py 10 29507 -l -d 0.0 --exp_id='ct28_agent10_right'"],

    # MIG 9 (1-8)
    #['MIG-bc91396f-1c2b-5319-8862-6f16d089ce5e', "python run_mctgraph.py 11 29508 -l -d 0.0 --exp_id='ct28_agent11_right'"],

    # MIG 10 (1-9)
    #['MIG-13d6b3aa-c302-5ac2-9183-5494d22547e6', "python run_mctgraph.py 12 29509 -l -d 0.0 --exp_id='ct28_agent12_right'"],

    # MIG 11 (1-10)
    #['MIG-f2d8b14c-d00c-5b5f-be98-bd1bff3bf371', "python run_mctgraph.py 13 29510 -l -d 0.0 --exp_id='ct28_agent13_right'"],
    
    # MIG 12 (1-11)
    #['MIG-62fabbf0-b8de-5040-b4db-93f62c477543', "python run_mctgraph.py 14 29511 -l -d 0.0 --exp_id='ct28_agent14_right'"],

    # MIG 13 (1-12)
    #['MIG-fcda8ac4-e82d-5259-8495-c777d8c95d74', "python run_mctgraph.py 15 29512 -l -d 0.0 --exp_id='ct28_agent15_right'"],

    # MIG 14 (1-13)
    #['MIG-f4aecf22-d8a0-50a9-b804-c65cb2d2ff7e', "python run_mctgraph.py 16 29513 -l -d 0.0 --exp_id='ct28_agent16_right'"]


    # EVALUATION AGENT
    # MIG 14 (1-13)
    #['MIG-f4aecf22-d8a0-50a9-b804-c65cb2d2ff7e', "python run_mctgraph.py 26 29513 -l -e --exp_id='ct28_md_eval_full' --reference='reference_eval.csv'"]
]

commands_ctgraph_full = [
    # IMG SEED 1 AGENTS
    # MIG 1 (0-7)
    ['MIG-c8cbc779-6499-5d1f-86e7-ad789e047331', "python run_mctgraph.py 0 29500 -l -d 0.0 --exp_id='ct28_agent0_left'"],
    ['MIG-c8cbc779-6499-5d1f-86e7-ad789e047331', "python run_mctgraph.py 1 29501 -l -d 0.0 --exp_id='ct28_agent1_left'"],

    # MIG 2 (0-8)
    ['MIG-f6c4eed6-1edf-5515-8e0b-0a5521e81cbb', "python run_mctgraph.py 2 29502 -l -d 0.0 --exp_id='ct28_agent2_left'"],
    ['MIG-f6c4eed6-1edf-5515-8e0b-0a5521e81cbb', "python run_mctgraph.py 3 29503 -l -d 0.0 --exp_id='ct28_agent3_left'"],

    # MIG 3 (0-9)
    ['MIG-ae777ac5-da68-5606-8581-1878288224dc', "python run_mctgraph.py 4 29504 -l -d 0.0 --exp_id='ct28_agent4_left'"],
    ['MIG-ae777ac5-da68-5606-8581-1878288224dc', "python run_mctgraph.py 5 29505 -l -d 0.0 --exp_id='ct28_agent5_left'"],

    # MIG 4 (0-10)
    ['MIG-b2a723b5-db7a-50d8-a267-70569dafe609', "python run_mctgraph.py 6 29506 -l -d 0.0 --exp_id='ct28_agent6_left'"],
    ['MIG-b2a723b5-db7a-50d8-a267-70569dafe609', "python run_mctgraph.py 7 29507 -l -d 0.0 --exp_id='ct28_agent7_left'"],

    # MIG 5 (0-11)
    ['MIG-73000f9d-03d4-5dab-ad23-34b546569bdd', "python run_mctgraph.py 8 29508 -l -d 0.0 --exp_id='ct28_agent8_left'"],
    ['MIG-73000f9d-03d4-5dab-ad23-34b546569bdd', "python run_mctgraph.py 9 29509 -l -d 0.0 --exp_id='ct28_agent9_left'"],

    # MIG 6 (0-12)
    #['MIG-f86cf155-b80e-56da-92ed-03c78bf647c7', "python run_mctgraph.py 5 29505 -l -d 0.0 --exp_id='ct28_agent5_left'"],

    # MIG 7 (0-13)
    #['MIG-bfb3fc5a-8b5e-59c1-87e8-3468c995103d', "python run_mctgraph.py 6 29506 -l -d 0.0 --exp_id='ct28_agent6_left'"],


    # IMG SEED 2 AGENTS
    # MIG 8 (1-7)
    ['MIG-f61216a8-2a31-500e-acf7-f0158fbf7ce3', "python run_mctgraph.py 10 29510 -l -d 0.0 --exp_id='ct28_agent10_right'"],
    ['MIG-f61216a8-2a31-500e-acf7-f0158fbf7ce3', "python run_mctgraph.py 11 29511 -l -d 0.0 --exp_id='ct28_agent11_right'"],

    # MIG 9 (1-8)
    ['MIG-bc91396f-1c2b-5319-8862-6f16d089ce5e', "python run_mctgraph.py 12 29512 -l -d 0.0 --exp_id='ct28_agent12_right'"],
    ['MIG-bc91396f-1c2b-5319-8862-6f16d089ce5e', "python run_mctgraph.py 13 29513 -l -d 0.0 --exp_id='ct28_agent13_right'"],

    # MIG 10 (1-9)
    ['MIG-13d6b3aa-c302-5ac2-9183-5494d22547e6', "python run_mctgraph.py 14 29514 -l -d 0.0 --exp_id='ct28_agent14_right'"],
    ['MIG-13d6b3aa-c302-5ac2-9183-5494d22547e6', "python run_mctgraph.py 15 29515 -l -d 0.0 --exp_id='ct28_agent15_right'"],

    # MIG 11 (1-10)
    ['MIG-f2d8b14c-d00c-5b5f-be98-bd1bff3bf371', "python run_mctgraph.py 16 29516 -l -d 0.0 --exp_id='ct28_agent16_right'"],
    ['MIG-f2d8b14c-d00c-5b5f-be98-bd1bff3bf371', "python run_mctgraph.py 17 29517 -l -d 0.0 --exp_id='ct28_agent17_right'"],
    
    # MIG 12 (1-11)
    ['MIG-62fabbf0-b8de-5040-b4db-93f62c477543', "python run_mctgraph.py 18 29518 -l -d 0.0 --exp_id='ct28_agent18_right'"],
    ['MIG-62fabbf0-b8de-5040-b4db-93f62c477543', "python run_mctgraph.py 19 29519 -l -d 0.0 --exp_id='ct28_agent19_right'"],

    # MIG 13 (1-12)
    #['MIG-fcda8ac4-e82d-5259-8495-c777d8c95d74', "python run_mctgraph.py 15 29512 -l -d 0.0 --exp_id='ct28_agent15_right'"],

    # MIG 14 (1-13)
    #['MIG-f4aecf22-d8a0-50a9-b804-c65cb2d2ff7e', "python run_mctgraph.py 16 29513 -l -d 0.0 --exp_id='ct28_agent16_right'"]
]



# FOCCAL (Minigrid)
commands_minigrid = [
    # MIG 1 (0-7)
    ['MIG-c8cbc779-6499-5d1f-86e7-ad789e047331', "python run_minigrid.py 0 29500 -l -d 0.0 --exp_id='mg_agent1'"],

    # MIG 2 (0-8)
    ['MIG-f6c4eed6-1edf-5515-8e0b-0a5521e81cbb', "python run_minigrid.py 1 29501 -l -d 0.0 --exp_id='mg_agent2'"],

    # MIG 3 (0-9)
    ['MIG-ae777ac5-da68-5606-8581-1878288224dc', "python run_minigrid.py 2 29502 -l -d 0.0 --exp_id='mg_agent3'"],

    # MIG 4 (0-10)
    ['MIG-b2a723b5-db7a-50d8-a267-70569dafe609', "python run_minigrid.py 3 29503 -l -d 0.0 --exp_id='mg_agent4'"],

    # MIG 5 (0-11)
    ['MIG-73000f9d-03d4-5dab-ad23-34b546569bdd', "python run_minigrid.py 4 29504 -l -d 0.0 --exp_id='mg_agent5'"],

    # MIG 6 (0-12)
    ['MIG-f86cf155-b80e-56da-92ed-03c78bf647c7', "python run_minigrid.py 5 29505 -l -d 0.0 --exp_id='mg_agent6'"],

    # MIG 7 (0-13)
    ['MIG-bfb3fc5a-8b5e-59c1-87e8-3468c995103d', "python run_minigrid.py 6 29506 -l -d 0.0 --exp_id='mg_agent7'"],

    # MIG 8 (1-7)
    ['MIG-f61216a8-2a31-500e-acf7-f0158fbf7ce3', "python run_minigrid.py 7 29507 -l -d 0.0 --exp_id='mg_agent8'"],

    # MIG 9 (1-8)
    ['MIG-bc91396f-1c2b-5319-8862-6f16d089ce5e', "python run_minigrid.py 8 29508 -l -d 0.0 --exp_id='mg_agent9'"],

    # MIG 10 (1-9)
    ['MIG-13d6b3aa-c302-5ac2-9183-5494d22547e6', "python run_minigrid.py 9 29509 -l -d 0.0 --exp_id='mg_agent10'"],

    # MIG 11 (1-10)
    #['MIG-f2d8b14c-d00c-5b5f-be98-bd1bff3bf371', "python run_minigrid.py 10 29510 -l -d 0.0 --exp_id='mg_agent11'"],

    # MIG 12 (1-11)
    #['MIG-62fabbf0-b8de-5040-b4db-93f62c477543', "python run_minigrid.py 11 29511 -l -d 0.0 --exp_id='mg_agent12'"],

    # MIG 13 (1-12)
    #['MIG-fcda8ac4-e82d-5259-8495-c777d8c95d74', "python run_minigrid.py 12 29512 -l -d 0.0 --exp_id='mg_agent13'"],

    # MIG 14 (1-13)
    #['MIG-f4aecf22-d8a0-50a9-b804-c65cb2d2ff7e', "python run_minigrid.py 0 29513 -l -e --exp_id='mg_eval'"]
]


# FOCCAL (Procgen)
commands_procgen = [
    # MIG 1 (0-7)
    ['MIG-c8cbc779-6499-5d1f-86e7-ad789e047331', "python run_procgen.py 0 29500 -l -d 0.0 --exp_id='prc_agent1'"],

    # MIG 2 (0-8)
    ['MIG-f6c4eed6-1edf-5515-8e0b-0a5521e81cbb', "python run_procgen.py 1 29501 -l -d 0.0 --exp_id='prc_agent2'"],

    # MIG 3 (0-9)
    ['MIG-ae777ac5-da68-5606-8581-1878288224dc', "python run_procgen.py 2 29502 -l -d 0.0 --exp_id='prc_agent3'"],

    # MIG 4 (0-10)
    ['MIG-b2a723b5-db7a-50d8-a267-70569dafe609', "python run_procgen.py 3 29503 -l -d 0.0 --exp_id='prc_agent4'"],

    # MIG 5 (0-11)
    ['MIG-73000f9d-03d4-5dab-ad23-34b546569bdd', "python run_procgen.py 4 29504 -l -d 0.0 --exp_id='prc_agent5'"],

    # MIG 6 (0-12)
    ['MIG-f86cf155-b80e-56da-92ed-03c78bf647c7', "python run_procgen.py 5 29505 -l -d 0.0 --exp_id='prc_agent6'"],

    # MIG 7 (0-13)
    ['MIG-bfb3fc5a-8b5e-59c1-87e8-3468c995103d', "python run_minigrid.py 6 29506 -l -d 0.0 --exp_id='mg_agent7'"],

    # MIG 8 (1-7)
     ['MIG-f61216a8-2a31-500e-acf7-f0158fbf7ce3', "python run_minigrid.py 7 29507 -l -d 0.0 --exp_id='mg_agent8'"],

    # MIG 9 (1-8)
    ['MIG-bc91396f-1c2b-5319-8862-6f16d089ce5e', "python run_minigrid.py 8 29508 -l -d 0.0 --exp_id='mg_agent9'"],

    # MIG 10 (1-9)
    ['MIG-13d6b3aa-c302-5ac2-9183-5494d22547e6', "python run_minigrid.py 9 29509 -l -d 0.0 --exp_id='mg_agent10'"],

    # MIG 11 (1-10)
    #['MIG-f2d8b14c-d00c-5b5f-be98-bd1bff3bf371', "python run_minigrid.py 10 29510 -l -d 0.0 --exp_id='mg_agent11'"],

    # MIG 12 (1-11)
    #['MIG-62fabbf0-b8de-5040-b4db-93f62c477543', "python run_minigrid.py 11 29511 -l -d 0.0 --exp_id='mg_agent12'"],

    # MIG 13 (1-12)
    #['MIG-fcda8ac4-e82d-5259-8495-c777d8c95d74', "python run_minigrid.py 12 29512 -l -d 0.0 --exp_id='mg_agent13'"],

    # MIG 14 (1-13)
    #['MIG-f4aecf22-d8a0-50a9-b804-c65cb2d2ff7e', "python run_minigrid.py 0 29513 -l -e --exp_id='mg_eval'"]
]



parser = argparse.ArgumentParser()
parser.add_argument('--env', help='indicate which experiment is being run for command selection', type=str, default='ctgraph')
parser.add_argument('--exp', help='', type=str, default='')
args = parser.parse_args()
commands = None
if args.env == 'ctgraph':
    commands = commands_ctgraph
elif args.env == 'minigrid':
    commands = commands_minigrid
elif args.env == 'ctfull':
    commands = commands_ctgraph_full
elif args.env == 'procgen':
    commands = commands_procgen
else:
    raise ValueError(f'no commands have been setup for --exp {args.exp}')


env = dict(os.environ)


path_header = args.env
if len(args.exp) > 0:
    path_header = args.exp


# Run the commands in seperate terminals
processes = []
for command in commands:
    print(f"{command[0]}, {command[1]} -p {path_header}")
    env['CUDA_VISIBLE_DEVICES'] = command[0]
    process = subprocess.Popen(shlex.split(command[1] + f' -p {path_header}'), env=env)
    processes.append(process)
    #time.sleep(5)

for process in processes:
    stdout, stderr = process.communicate()