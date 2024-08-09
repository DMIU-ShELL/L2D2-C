import json
import os
import argparse

class Args():
    def __init__(self):
        self.general_seed = [3, 4, 5, 6, 7]
        self.depth = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.branching_factor = 2
        self.wait_prob = 0.0

        self.high_reward_value = 1.0
        self.crash_reward_value = 0.0
        self.stochastic_sampling = False
        self.reward_std = 0.1
        self.min_static_reward_episodes = 0
        self.max_static_reward_episodes = 0
        self.reward_distribution = "needle_in_haystack"

        self.MDP_decision_s = True
        self.MDP_wait_s = False
        #self.wait_states = [2,2]
        #self.decision_states = [3,5]
        #self.graph_ends = [6,9]

        self.seed = [
            [1, 2, 3, 4],
            [3, 4, 5, 6],
            [5, 6, 7, 8],
            [7, 8, 9, 10],
            [9, 10, 11, 12]
        ]
        self.oneD = False
        #self.nr_of_images = 10
        self.noise_on_images_on_read = 0
        self.small_rotation_on_read = 1

class CustomJSONEncoder(json.JSONEncoder):
    def __init__(self, *args, **kwargs):
        super(CustomJSONEncoder, self).__init__(*args, **kwargs)

    def encode(self, obj):
        if isinstance(obj, list):
            return '[' + ', '.join(json.dumps(el, cls=CustomJSONEncoder) for el in obj) + ']'
        return super(CustomJSONEncoder, self).encode(obj)

def generate_jsons(args):
    iterator = 0
    # Ensure the 'jsons' directory exists
    os.makedirs('jsons', exist_ok=True)
    
    for idx, general_seed in enumerate(args.general_seed):
        iterator += 1
        os.makedirs(f'jsons/seed{iterator}', exist_ok=True)
        for depth in args.depth:
            for image_seed in args.seed[idx]:
                a = (args.branching_factor**depth) + 1
                b = a+1
                c = a + (args.branching_factor**depth)

                wait_states = [2, 2]
                decision_states = [3, a]
                graph_ends = [b, c]
                nr_of_images = c+1

                js = {
                    "general_seed" : general_seed,                      # 1
                    "graph_shape" : {
                        "depth" : depth,                                # 2
                        "branching_factor" : args.branching_factor,
                        "wait_prob" : args.wait_prob
                    },
                    "reward" : {
                        "high_reward_value" : args.high_reward_value,
                        "crash_reward_value" : args.crash_reward_value,
                        "stochastic_sampling" : args.stochastic_sampling,
                        "reward_std" : args.reward_std,
                        "min_static_reward_episodes" : args.min_static_reward_episodes,
                        "max_static_reward_episodes" : args.max_static_reward_episodes,
                        "reward_distribution" : args.reward_distribution
                    },
                    "observations" : {
                        "MDP_decision_s" : args.MDP_decision_s,
                        "MDP_wait_s" : args.MDP_wait_s,
                        "wait_states" : wait_states,                    # 3
                        "decision_states" : decision_states,            # 4
                        "graph_ends" : graph_ends                       # 5
                    },
                    "image_dataset" : {
                        "seed" : image_seed,                            # 6
                        "1D" : args.oneD,
                        "nr_of_images" : nr_of_images,                  # 7
                        "noise_on_images_on_read" : args.noise_on_images_on_read,
                        "small_rotation_on_read" : args.small_rotation_on_read
                    }
                }
                name = f'ctgraph_d{depth}_imgseed{image_seed}_pomdp_wait'
                obj = json.dumps(js, indent=3, cls=CustomJSONEncoder)
                with open(f"jsons/seed{iterator}/{name}.json", "w") as outfile:
                    outfile.write(obj)

if __name__ == '__main__':
    args = Args()
    generate_jsons(args)

    
