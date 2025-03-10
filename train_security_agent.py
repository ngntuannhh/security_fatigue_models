import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
from security_env import SecurityEnv

# Check if tensorboard is installed
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Tensorboard is not installed. Training logs will not be saved.")
    print("To install tensorboard, run: pip install tensorboard")

def create_env(render_mode=None):
    """Create and wrap the environment."""
    env = SecurityEnv(render_mode=render_mode)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    return env

def train_agent():
    """Train the PPO agent on the SecurityEnv."""
    print("Starting training process...")
    
    # Create directories for saving models and logs
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Create training and evaluation environments
    train_env = create_env()
    eval_env = create_env()
    
    # Create the agent with optimized hyperparameters
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.01,
        tensorboard_log="./logs/" if TENSORBOARD_AVAILABLE else None,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[64, 64],
                vf=[64, 64]
            )
        ),
        verbose=1
    )
    
    # Create callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best_model",
        log_path="./logs",
        eval_freq=1000,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path="./models/",
        name_prefix="security_model"
    )
    
    # Train the agent
    total_timesteps = 200000  # Adjust based on your needs
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save the final model and normalization stats
    model.save("models/final_security_model")
    train_env.save("models/vec_normalize.pkl")
    
    print("\nTraining completed!")
    return model, train_env

def evaluate_agent(model, env, num_episodes=5):
    """Evaluate the trained agent."""
    print("\nEvaluating agent...")
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        truncated = False
        total_reward = 0
        episode_steps = 0
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            episode_steps += 1
            
            # Print step information
            print(f"\nEpisode {episode + 1}, Step {episode_steps}")
            print(f"Action: {action}")
            print(f"Reward: {reward:.2f}")
            print(f"Security Score: {info['security_score']:.2f}")
            print(f"Fatigue Score: {info['fatigue_score']:.2f}")
            print("Feature Values:")
            for name, value in info['feature_values'].items():
                print(f"  {name}: {value}")
        
        print(f"\nEpisode {episode + 1} completed:")
        print(f"Total Steps: {episode_steps}")
        print(f"Total Reward: {total_reward:.2f}")
        print("-" * 50)

def plot_training_results(log_dir="./logs"):
    """Plot the training results."""
    import pandas as pd
    
    # Read the training logs
    df = pd.read_csv(f"{log_dir}/monitor.csv", skiprows=1)
    
    # Plot episode rewards
    plt.figure(figsize=(12, 5))
    plt.plot(df['r'], label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Learning Curve")
    plt.legend()
    plt.savefig("logs/learning_curve.png")
    plt.close()
    
    # Plot moving average reward
    window_size = 10
    moving_avg = df['r'].rolling(window=window_size).mean()
    
    plt.figure(figsize=(12, 5))
    plt.plot(moving_avg, label=f"Moving Average Reward (window={window_size})", color='orange')
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Moving Average of Episode Rewards")
    plt.legend()
    plt.savefig("logs/moving_average.png")
    plt.close()

if __name__ == "__main__":
    # Train the agent
    model, env = train_agent()
    
    # Evaluate the agent
    eval_env = create_env(render_mode="human")
    evaluate_agent(model, eval_env)
    
    # Plot training results
    plot_training_results() 