import gym
from PIL import Image
import minihack
from nle import nethack
import argparse

# Function to convert MiniGrid environment observation to RGB image
def minihack_to_image(observation):
    # Check if the observation is a tuple
    if isinstance(observation, tuple):
        # If observation is a tuple, extract the first element
        img_data = observation[0]['pixel_crop']
    else:
        # If observation is a dictionary, extract the 'image' key
        img_data = observation['pixel_crop']
    
    # Convert image data to a PIL Image
    img = Image.fromarray(img_data)
    
    return img

# Function to save MiniGrid environment as PNG image
def save_minihack_image(env_name, filename, seed):  
    # Create MiniGrid environment

    MOVE_ACTIONS = tuple(nethack.CompassDirection)
    NAVIGATE_ACTIONS = MOVE_ACTIONS + (
        nethack.Command.OPEN,
        nethack.Command.KICK,
    )
    env = gym.make(env_name, observation_keys=("pixel_crop", "glyphs"), actions=NAVIGATE_ACTIONS)
    env.seed(seed)
    
    # Reset environment to get initial observation
    obs = env.reset()

    glyphs = obs["glyphs"]
    pixel_crop = obs["pixel_crop"]
    
    # Save image as PNG
    img.save(filename)
    
    # Close environment
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-envs', help='', nargs='+')
    args = parser.parse_args()

    # Example usage:
    seed = 840
    for env_name in args.envs:   # Change this to the MiniGrid environment you want
        filename = 'minigrid_images/' + str(env_name) + str(seed) + '.png'      # Specify the filename for the saved image
        save_minihack_image(env_name, filename, seed)
