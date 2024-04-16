import gym
from PIL import Image
from gym_minigrid.wrappers import ReseedWrapper, ImgObsWrapper, RGBImgObsWrapper
import argparse

# Function to convert MiniGrid environment observation to RGB image
def minigrid_to_image(observation):
    # Check if the observation is a tuple
    if isinstance(observation, tuple):
        # If observation is a tuple, extract the first element
        img_data = observation[0]['image']
    else:
        # If observation is a dictionary, extract the 'image' key
        img_data = observation['image']
    
    # Convert image data to a PIL Image
    img = Image.fromarray(img_data)
    
    return img

# Function to save MiniGrid environment as PNG image
def save_minigrid_image(env_name, filename, seed):
    # Create MiniGrid environment
    env = gym.make(env_name)

    env = RGBImgObsWrapper(env)

    #envs_list = [env1, env2, env3]
    #envs_list = [ReseedWrapper(env, (seed,)) for env, seed in zip(envs_list, [129,112,237])]

    
    env = ReseedWrapper(env, (seed,))
    
    # Reset environment to get initial observation
    obs = env.reset()
    
    # Convert observation to RGB image
    img = minigrid_to_image(obs)
    
    # Save image as PNG
    img.save(filename)
    
    # Close environment
    env.close()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-envs', help='', nargs='+')
    args = parser.parse_args()

    # Example usage:
    seed = 9157
    for env_name in args.envs:   # Change this to the MiniGrid environment you want
        filename = 'minigrid_images/' + str(env_name) + str(seed) + '.png'      # Specify the filename for the saved image
        save_minigrid_image(env_name, filename, seed)
