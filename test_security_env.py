import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import matplotlib.pyplot as plt
from security_env import SecurityEnv
import os

def print_feature_info(env):
    """Print information about the features in the environment."""
    print("\nFeature Information:")
    print("-" * 50)
    for name in env.feature_names:
        feature_type = env.feature_types.get(name, 'unknown')
        if feature_type == 'categorical':
            print(f"{name}:")
            print(f"  Type: {feature_type}")
            print(f"  Values: 0 (Beginner), 1 (Intermediate), 2 (Advanced)")
        elif feature_type == 'ordinal':
            print(f"{name}:")
            print(f"  Type: {feature_type}")
            print(f"  Range: {env.feature_ranges.get(name, (1, 5))}")
            print(f"  Values: 1 (Lowest) to 5 (Highest)")
        elif feature_type == 'binary':
            print(f"{name}:")
            print(f"  Type: {feature_type}")
            print(f"  Values: 0 (No) or 1 (Yes)")
        else:
            print(f"{name}:")
            print(f"  Type: {feature_type}")
    print("-" * 50)

def test_environment():
    """Test the basic functionality of the SecurityEnv."""
    print("Testing Security Environment...")
    
    # Check if model file exists
    model_path = "fatigue_model.joblib"
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Please make sure the Random Forest model is saved as 'fatigue_model.joblib'")
        return
    
    # Create environment
    env = SecurityEnv(render_mode="human")
    
    # Print feature information
    print_feature_info(env)
    
    print("\nTesting environment reset:")
    obs, info = env.reset()
    print("Initial observation shape:", obs.shape)
    
    # Print initial state
    print("\nInitial State:")
    print("-" * 50)
    print("Feature Values:")
    for name, value in info['feature_values'].items():
        feature_type = env.feature_types.get(name, 'unknown')
        if feature_type == 'categorical':
            level = ["Beginner", "Intermediate", "Advanced"][int(value)]
            print(f"  {name}: {value} ({level})")
        elif feature_type == 'ordinal':
            print(f"  {name}: {value}")
        elif feature_type == 'binary':
            print(f"  {name}: {value} ({'Yes' if value == 1 else 'No'})")
    
    print(f"\nSecurity Score: {info['security_score']:.2f}")
    print(f"Fatigue Score: {info['fatigue_score']:.2f}")
    print("-" * 50)
    
    # Test a few random steps
    print("\nTesting random steps:")
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"\nStep {i+1}:")
        print("-" * 50)
        print("Action:", action)
        print(f"Reward: {reward:.2f}")
        print("\nFeature Values:")
        for name, value in info['feature_values'].items():
            feature_type = env.feature_types.get(name, 'unknown')
            if feature_type == 'categorical':
                level = ["Beginner", "Intermediate", "Advanced"][int(value)]
                print(f"  {name}: {value} ({level})")
            elif feature_type == 'ordinal':
                print(f"  {name}: {value}")
            elif feature_type == 'binary':
                print(f"  {name}: {value} ({'Yes' if value == 1 else 'No'})")
        print(f"\nSecurity Score: {info['security_score']:.2f}")
        print(f"Fatigue Score: {info['fatigue_score']:.2f}")
        env.render()

def train_agent():
    """Train a PPO agent on the SecurityEnv."""
    print("\nTraining RL Agent...")
    
    # Check if model file exists
    model_path = "fatigue_model.joblib"
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Please make sure the Random Forest model is saved as 'fatigue_model.joblib'")
        return
    
    # Create and wrap the environment
    env = SecurityEnv()
    env = DummyVecEnv([lambda: env])
    
    # Create the agent
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1
    )
    
    # Create evaluation callback
    eval_env = SecurityEnv()
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model",
        log_path="./logs",
        eval_freq=1000,
        deterministic=True,
        render=False
    )
    
    # Train the agent
    total_timesteps = 100000  # Adjust based on your needs
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True
    )
    
    # Save the final model
    model.save("final_security_model")
    
    # Test the trained agent
    print("\nTesting trained agent:")
    test_env = SecurityEnv(render_mode="human")
    obs, info = test_env.reset()
    done = False
    truncated = False
    total_reward = 0
    
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)
        total_reward += reward
        test_env.render()
    
    print(f"\nTotal reward: {total_reward:.2f}")

def plot_training_results(log_dir="./logs"):
    """Plot the training results from the logs."""
    import pandas as pd
    
    # Check if log directory exists
    if not os.path.exists(log_dir):
        print(f"Error: Log directory '{log_dir}' not found!")
        print("Please make sure the training has been completed first.")
        return
    
    # Read the training logs
    log_file = f"{log_dir}/monitor.csv"
    if not os.path.exists(log_file):
        print(f"Error: Log file '{log_file}' not found!")
        print("Please make sure the training has been completed first.")
        return
    
    df = pd.read_csv(log_file, skiprows=1)
    
    # Plot episode rewards
    plt.figure(figsize=(12, 5))
    plt.plot(df['r'], label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Learning Curve")
    plt.legend()
    plt.show()
    
    # Plot moving average reward
    window_size = 10
    moving_avg = df['r'].rolling(window=window_size).mean()
    
    plt.figure(figsize=(12, 5))
    plt.plot(moving_avg, label=f"Moving Average Reward (window={window_size})", color='orange')
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Moving Average of Episode Rewards")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Test the environment
    test_environment()
    
    # Train the agent
    train_agent()
    
    # Plot training results
    plot_training_results() 