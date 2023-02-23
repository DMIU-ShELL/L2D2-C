import json
import random
import matplotlib.pyplot as plt
import numpy as np

NUM_TASKS = 16
NUM_AGENTS = 12
CYCLES = 4



random.seed(9157)
z = []
y = {"seed" : [], "agents" : []}



# SEED
theone = []
for _ in range(NUM_AGENTS):
    theone.append(random.randint(1000, 9999))
y["seed"] = theone


# Evaluation agent
y["agents"].append({"task_ids" : list(list(range(0, NUM_TASKS))*CYCLES), "max_steps" : 12800})


# CURRICULUM
for _ in range(NUM_AGENTS):
    theone = []
    for _ in range(CYCLES * NUM_TASKS):
        theone.append(random.randint(0, NUM_TASKS-1))
    z.append(list(theone))
    y["agents"].append({"task_ids" : list(theone), "max_steps" : 12800})


# HISTOGRAM
z_ = [j for sub in z for j in sub]
counts, edges, bars = plt.hist(z_, bins=NUM_TASKS)
plt.bar_label(bars)
plt.xlabel('Task')
plt.ylabel('Occurences')
plt.savefig('curriculum_hist.pdf', dpi=300)


# IMAGE
plt.clf()
plt.imshow(z, cmap='viridis')
plt.colorbar(orientation='horizontal')
plt.savefig('curriculum_matrix.pdf', dpi=1000, bbox_inches='tight')


# WRITE TO FILE
json_obj = json.dumps(y, indent=4)
with open("curriculum.json", "w") as f:
    f.write(json_obj)