import json
import random
import matplotlib.pyplot as plt
import numpy as np

NUM_TASKS = 16
NUM_AGENTS = 13
CYCLES = 4



random.seed(9157)
z = []
y = {"seed" : [], "agents" : []}



theone = []
for _ in range(NUM_AGENTS):
    theone.append(random.randint(1000, 9999))

y["seed"] = theone

x = list(range(NUM_TASKS))
theone = []
for _ in range(CYCLES):
    theone.extend(list(x))
z.append(list(theone))
y["agents"].append({"task_ids" : list(theone), "max_steps" : 12800})



for _ in range(NUM_AGENTS):
    theone = []
    for _ in range(CYCLES):
        random.shuffle(x)   # randomizer
        theone.extend(list(x))
    random.shuffle(theone)
    z.append(list(theone))
    y["agents"].append({"task_ids" : list(theone), "max_steps" : 12800})


z_ = [item for sublist in z for item in sublist]
plt.hist(z_, bins=NUM_TASKS)
plt.savefig('curriculum_hist.pdf', dpi=300)

plt.clf()

plt.imshow(z, cmap='viridis')
plt.colorbar(orientation='horizontal')
plt.savefig('curriculum_matrix.pdf', dpi=1000, bbox_inches='tight')


json_obj = json.dumps(y, indent=4)


with open("curriculum.json", "w") as f:
    f.write(json_obj)