import pandas as pd




ll_paths = ['log/2x2/llagent/eval_metrics_agent_0.csv', 'log/4x4/llagent/eval_metrics_agent_0.csv', 'log/8x8/llagent/eval_metrics_agent_0.csv', 'log/16x16/llagent/eval_metrics_agent_0.csv' ,'log/32x32/llagent/eval_metrics_agent_0.csv']
shell_paths = ['log/2x2/shell/eval_metrics_agent_0.csv', 'log/4x4/shell/eval_metrics_agent_0.csv', 'log/8x8/shell/eval_metrics_agent_0.csv', 'log/16x16/shell/eval_metrics_agent_0.csv', 'log/32x32/shell/eval_metrics_agent_0.csv']
ll_ratio = [5.22, 2.41, 2.6275, 2.46, 2.03]
shell_ratio = [3.17, 1.29, 0.97, 0.7, 1.603515625]

for i in range(len(ll_paths)):
    ll = pd.read_csv(ll_paths[i])
    shell = pd.read_csv(shell_paths[i])

    split_ll = ll_paths[i].split('/')
    split_shell = shell_paths[i].split('/')
    
    name = '/home/lunet/cosn5/working/ctgraph/DeepRL-0.3/new_eval_metrics/' + split_ll[1] + '/ll/eval_metrics_agent_0.csv'
    ll_new = ll.iloc[::round(ll_ratio[i])]
    ll_new.to_csv(name, index=False)


    name = '/home/lunet/cosn5/working/ctgraph/DeepRL-0.3/new_eval_metrics/' + split_shell[1] + '/shell/eval_metrics_agent_0.csv'
    shell_new = shell.iloc[::round(shell_ratio[i])]
    shell_new.to_csv(name, index=False)