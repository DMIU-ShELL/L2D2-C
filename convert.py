import pandas as pd

fpaths = ['log/2x2/llagent/MetaCTgraph-shell-eval-upz-seed-9157/agent_0/221101-132505/eval_metrics_agent_0.csv',
          'log/4x4/llagent/MetaCTgraph-shell-eval-upz-seed-9157/agent_0/221101-133813/eval_metrics_agent_0.csv',
          'log/8x8/llagent/MetaCTgraph-shell-eval-upz-seed-9157/agent_0/221101-141118/eval_metrics_agent_0.csv',
          'log/16x16/llagent/MetaCTgraph-shell-eval-upz-seed-9157/agent_0/221101-162046/eval_metrics_agent_0.csv',
          'log/32x32/llagent/MetaCTgraph-shell-eval-upz-seed-9157/agent_0/221102-112438/eval_metrics_agent_0.csv']


for path in fpaths:
    broken = path.split('/')


    df = pd.read_csv(path)
    result = pd.to_datetime(df.iloc[:, 36], unit='s')
    df['datetime'] = result

    df.to_csv(broken[1] + '.csv', index=False)