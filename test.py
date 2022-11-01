import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

df = pd.read_csv('log/16x16_shell/MetaCTgraph-shell-dist-upz-seed-9158/agent_1/221030-111703/task_changes_1.csv')
print(df)

#fig = df.plot(kind='barh', stacked=True)
agents = ['JOB1','JOB2','JOB3','JOB4']

# input wait times
changetimes = ['03:20:50','04:45:10','06:10:40','05:30:30']

num_tasks = len(df.index)
# converting wait times to float
waittimes = []
for wt in changetimes:
    waittime = datetime.strptime(wt,'%H:%M:%S')
    waittime = waittime.hour + waittime.minute/60 + waittime.second/3600
    waittimes.append(waittime)

# input run times
runtimesin = ['00:20:50','01:00:10','00:30:40','00:10:30']
# converting run times to float    
runtimes = []
for rt in runtimesin:
    runtime = datetime.strptime(rt,'%H:%M:%S')
    runtime = runtime.hour + runtime.minute/60 + runtime.second/3600
    runtimes.append(runtime)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.barh(agents, waittimes, align='center', height=.25, color='#00ff00',label='wait time')
ax.barh(agents, runtimes, align='center', height=.25, left=waittimes, color='g',label='run time')
ax.set_yticks(agents)
ax.set_xlabel('Hour')
ax.set_title('Run Time by Job')
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.savefig('example.pdf', dpi=256, format='pdf')