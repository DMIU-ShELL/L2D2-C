import pandas as pd
import matplotlib.pyplot as plt
import os

rootdir = '/home/lunet/cosn5/working/ctgraph/DeepRL-0.3/log/kill_me/2x2/shell'

dfs = []
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if 'timings' in file:
            if file != 'timings0.csv' or file != 'timings33.csv':
                path = os.path.join(subdir, file)
                df = pd.read_csv(path)
                dfs.append(df)

avg = pd.DataFrame()
for index, val in enumerate(dfs):
    avg['agent'+str(index)] = val.iloc[:, 5]

avg['avg'] = avg.mean(axis=1)

fig = avg['avg'].plot.line(figsize=(50, 10), )

for label in (fig.get_xticklabels() + fig.get_yticklabels()):
    label.set_fontsize(25)

plt.xlabel('Communication loops/iteration', fontsize=25)
plt.ylabel('Time in seconds', fontsize=25)
plt.title('Average time taken per communication loop/iteration', fontsize=25)
plt.grid()
plt.savefig('timings.pdf', dpi=256, bbox_inches='tight')
