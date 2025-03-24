import gymnasium as gym
import torch
import imageio
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

wandb.init(project="algorithmExploration", entity="olivarad")

# Define metrics
wandb.define_metric("agent")
wandb.define_metric("agent/meanReward")
wandb.define_metric("agent/final_STDDev")
wandb.define_metric("configuration/learningRate")
wandb.define_metric("configuration/batch_size")
wandb.define_metric("configuration/gamma")
wandb.define_metric("configuration/total_timesteps")

hyperparameters = {
    "learning_rate": 3e-4,
    "batch_size": 64,
    "gamma": 0.99,
    "total_timesteps": 500000
}

wandb.config.update(hyperparameters)
wandb.log({
    "configuration/learningRate": hyperparameters["learning_rate"],
    "configuration/batch_size": hyperparameters["batch_size"],
    "configuration/gamma": hyperparameters["gamma"],
    "configuration/total_timesteps": hyperparameters["total_timesteps"],
}, step=1)

# Define environment
env = gym.make('CartPole-v1', render_mode='rgb_array')
env = Monitor(env)

# PPO Hyperparameters
policy_kwargs = dict(
    net_arch=[256, 256],
    activation_fn=torch.nn.ReLU
)

# Initialize the PPO model
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=wandb.config["learning_rate"],
    batch_size=wandb.config["batch_size"],
    gamma=wandb.config["gamma"],
    n_steps=2048,
    ent_coef=0.01,
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log="./ppo_cartpole"
)

# Define WandB callback
class WandBTrainingCallback(BaseCallback):
    def __init__(self, log_interval=1000, verbose=1):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.episode_rewards = []
        self.episode_lengths = []
        self.total_steps = 0

    def _on_step(self) -> bool:
        if "episode" in self.locals:
            self.episode_rewards.append(self.locals["episode"]["r"])
            self.episode_lengths.append(self.locals["episode"]["l"])
        
        self.total_steps += 1
        if self.total_steps % self.log_interval == 0:
            mean_reward = sum(self.episode_rewards) / len(self.episode_rewards) if self.episode_rewards else 0
            mean_length = sum(self.episode_lengths) / len(self.episode_lengths) if self.episode_lengths else 0

            wandb.log({
                "training/total_steps": self.total_steps,
                "training/mean_reward": mean_reward,
                "training/mean_length": mean_length
            })
            self.episode_rewards = []
            self.episode_lengths = []
        
        return True

# Train the model
model.learn(total_timesteps=wandb.config["total_timesteps"], callback=WandBTrainingCallback())

# Save the trained model
model.save("ppo_cartpole")

# Evaluate the trained model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
wandb.log({
    "agent/meanReward": mean_reward,
    "agent/final_STDDev": std_reward
})
print(f"Mean Reward: {mean_reward}, STD Dev: {std_reward}")

# Capture a 30-second GIF of the trained model in action
frames = []
obs, info = env.reset()
done = False
frame_count = 0
max_frames = 30 * 60

while frame_count < max_frames:
    frame_count += 1
    action, _states = model.predict(obs)  # Get the agent's action
    obs, reward, done, truncated, info = env.step(action)  # Take a step in the environment

    # Capture frame and append
    frame = env.render()
    frames.append(frame)

    if done or truncated:
        break

# Save the frames as a GIF
output_gif_path = "cartpole_ppo.gif"
imageio.mimsave(output_gif_path, frames, duration=1/60)

# Upload GIF to WandB
wandb.log({"training_gif": wandb.Video(output_gif_path, format="gif")})

# Close the environment
env.close()

# Finish the WandB run
wandb.finish()