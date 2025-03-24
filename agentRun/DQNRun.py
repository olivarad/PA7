import gymnasium as gym
from stable_baselines3 import DQN
import imageio
import numpy as np

# Load the trained model
model = DQN.load("dqn_cartpole.zip")

# Create the environment
env = gym.make('CartPole-v1', render_mode='rgb_array')  # 'rgb_array' allows capturing frames

# Initialize the GIF writer
frames = []

# Run the trained agent in the environment
obs, info = env.reset()  # Get the initial observation (returns a tuple)
done = False
frame_count = 0
max_frames = 30 * 60  # 30 seconds, assuming 60 frames per second (FPS)

while not done and frame_count < max_frames:
    frame_count += 1
    action, _states = model.predict(obs)  # Get the agent's action
    obs, reward, done, truncated, info = env.step(action)  # Take a step in the environment
    frame = env.render()  # Render and capture the current frame

    frames.append(frame)  # Store the frame

# Save the frames as a GIF
output_gif_path = "cartpole_dqn.gif"
imageio.mimsave(output_gif_path, frames, duration=1/60)  # Save as GIF with ~60 FPS

env.close()  # Close the environment after running
print(f"GIF saved as {output_gif_path}")