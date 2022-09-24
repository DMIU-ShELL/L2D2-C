import multiprocessing as mp
import time


class Communication(object):
    mask = 0
    def __init__(self, agent_id):
        self.agent_id = agent_id

    def send_mask(self, task_label):
        return mask

    def recieve_mask(self, mask):
        self.mask = mask

    def send_mask_mp(self, task_label, masks_recieved):
        if task_label != None:
            #print(mask)
            masks_recieved.put(mask)
        
    def recieve_mask_mp(self, masks_recieved):
        if not masks_recieved.empty():
            mask = masks_recieved.get() + 1
            #print(mask)
            print('***LOGGER*** Mask value is: ' + str(mask), flush=True)

        else:
            print('--- NO MASK IN QUEUE ---', flush=True)
        

class Agent(object):
    def __init__(self, mask):
        self.mask = mask
        self.reward = 1

    def iteration(self):
        return self.reward

    def iteration_mp(self, rewards):
        rewards.put(self.reward)
        

def train(comm, agent, task_label):
    episode_reward = 0
    
    while task_label < 69:
        task_label += 1

        if task_label % 2 == 0:
            reward = agent.iteration()
            episode_reward += reward
            print('***LOGGER*** Episodic reward is: ' + str(episode_reward))

            
            mask = comm.send_mask(task_label)
            #mask = agent.increment_mask(mask)
            comm.recieve_mask(mask)
            print('***LOGGER*** Mask value is: ' + str(mask))


def train_mp(comm, agent, task_label):
    mp.set_start_method('fork')
    rewards = mp.Queue()
    masks_recieved = mp.Queue()
    masks_updated = mp.Queue()

    rewards_conn_a, rewards_conn_b = Pipe()

    episode_reward = 0

    
    while task_label < 69:
        task_label += 1

        if task_label % 2 == 0:
            a1 = mp.Process(target=agent.iteration_mp, args=(rewards, ))
            a1.start()
            #a1.join()
            #reward = agent.iteration()
            while not rewards.empty():
                episode_rewards += rewards_conn_b.recv()
                episode_reward += rewards.get()
            print('***LOGGER*** Episodic reward is: ' + str(episode_reward))


            c1 = mp.Process(target=comm.send_mask_mp, args=(task_label, masks_recieved))
            c1.start()
            #c1.join()

            c2 = mp.Process(target=comm.recieve_mask_mp, args=(masks_recieved,))
            c2.start()
            #c2.join()
                
            


if __name__ == '__main__':
    start_time = time.time()
    task_label = 0
    mask = 0
    agent_id = 0

    comm = Communication(agent_id)
    agent = Agent(mask)

    #train(comm, agent, task_label)
    train_mp(comm, agent, task_label)

    print('--- %s seconds ---' % (time.time() - start_time))
