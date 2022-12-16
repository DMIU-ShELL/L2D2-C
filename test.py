from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import os
import re


rootdir = 'log'

# Load the data
data = {}
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if 'task_changes' in file:
            path = os.path.join(subdir, file)
            path_s = path.split('/')
            agent = int([s for s in path_s if "agent" in s][0].split('_')[1])
            df = pd.read_csv(path)
            data[agent] = df

for key, val in sorted(data.items()):




fig = plt.figure()
ax = fig.add_subplot(111)
for id, df in sorted(data.items()):
    # input wait times
    task_changes = ['03:20:50','04:45:10','06:10:40','05:30:30']
    # converting wait times to float
    runtimes = []
    for tc in task_changes:
        waittime = datetime.strptime(tc,'%H:%M:%S')
        waittime = waittime.hour + waittime.minute/60 + waittime.second/3600
        runtimes.append(waittime)

    ax.barh(data.keys(), runtimes, align='center', height=.25, left=runtimes, color='g',label='run time')

#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.barh(data.keys(), waittimes, align='center', height=.25, color='#00ff00',label='wait time')
#ax.barh(data.keys(), runtimes, align='center', height=.25, left=waittimes, color='g',label='run time')
ax.set_yticks(data.keys())
ax.set_xlabel('')
ax.set_title('')
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.savefig('curriculum.pdf')