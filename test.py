import numpy as np
import torch

mask_rewards_dict = {(1.0, 0.0, 0.0): 0.99, (0.0, 1.0, 0.0): 0.98, (0.0, 0.0, 1.0): 0.97}
#mask_rewards_dict = {(1.0, 0.0, 0.0): 0.99, (0.0, 1.0, 0.0): 0.0, (0.0, 0.0, 1.0): 0.0}
#mask_rewards_dict = {(1.0, 0.0, 0.0): 0.0}
#mask_rewards_dict = {}
#print(mask_rewards_dict)


req_label_as_np1 = torch.tensor([0., 1., 0.]).detach().cpu().numpy()
req_label_as_np2 = torch.tensor([0., 0., 1.]).detach().cpu().numpy()
a = [req_label_as_np1, req_label_as_np2]
b = [1, 2]


meta_responses = []
expecting = []
if a:
    for idx, val in enumerate(a):
        req_label_as_np = a[idx]
        req_id = b[idx]
        print(req_id, req_label_as_np)


        d = {}
        print('Knowledge base', mask_rewards_dict)
        for tlabel, treward in mask_rewards_dict.items():
            if treward != 0.0:
                print(np.asarray(tlabel), treward)
                dist = np.sum(abs(np.subtract(req_label_as_np, np.asarray(tlabel))))
                if dist <= 0.0:
                    d['dst_agent_id'] = req_id
                    d['mask_reward'] = treward
                    d['dist'] = dist
                    d['resp_task_label'] = torch.tensor(tlabel)
                    expecting.append(d['dst_agent_id'])
                    meta_responses.append(d)
                    
        if not d:
            d['dst_agent_id'] = req_id
            d['mask_reward'] = torch.inf
            d['dist'] = torch.inf
            d['resp_task_label'] = torch.tensor([torch.inf] * 3)

            #expecting.append(d['dst_agent_id'])
            meta_responses.append(d)

print(meta_responses)
