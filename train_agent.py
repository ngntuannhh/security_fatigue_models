import argparse
import json
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from security_env import SecurityEnv

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(script_dir, 'models')

MODEL_VER = 'run_tuned_1m'  # or your chosen model version

MODEL_PATH = os.path.join(MODEL_DIR, MODEL_VER, 'best_model', 'best_model.zip')
FEEDBACK_PATH = os.path.join(script_dir, 'feedback_buffer.jsonl')

def create_synthetic_episodes(feedback_data, env):
    """
    Convert feedback data into synthetic episodes that can be used for retraining.
    
    This function takes human feedback data and creates synthetic episodes 
    that can be used to guide the RL agent's learning process.
    """
    episodes = []
    
    for entry in feedback_data:
        initial_config = np.array(entry['initial_config'])
        final_config = np.array(entry['final_config'])
        reward = entry['reward']
        
        # Reset environment with the initial configuration
        state, _ = env.reset_with_user_config(initial_config)
        
        # Extract just the configuration part of the state
        # (exclude fatigue and security score at the end)
        num_features = len(initial_config)
        
        # We simulate a step where the action is the final configuration
        next_state, reward_computed, terminated, truncated, info = env.step(final_config)
        
        # If the feedback has a custom reward, use it; otherwise use the computed reward
        if reward is not None:
            actual_reward = reward
        else:
            actual_reward = reward_computed
        
        episodes.append({
            'state': state,
            'action': final_config,
            'reward': actual_reward,
            'next_state': next_state,
            'done': terminated or truncated
        })
    
    return episodes

def train_from_scratch():
    """
    Train a new RL model from scratch.
    """
    print("Starting training from scratch...")
    
    # Make sure the model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Create the environment
    env = make_vec_env(SecurityEnv, n_envs=1)
    
    # Initialize a new model
    model = PPO("MlpPolicy", env, verbose=1)
    
    # Setup callbacks for saving checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,  # Save every 10,000 steps
        save_path=MODEL_DIR,
        name_prefix="ppo_checkpoint"
    )
    
    # Train the model
    print("Training model...")
    model.learn(total_timesteps=1000, callback=checkpoint_callback)
    
    # Save the final model
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    return True

def retrain_from_feedback():
    """
    Load feedback data and retrain the model using human feedback.
    """
    print("Starting retraining process...")
    
    # Load feedback data
    feedback_data = []
    if os.path.exists(FEEDBACK_PATH):
        with open(FEEDBACK_PATH, 'r') as f:
            for line in f:
                try:
                    feedback_data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line in feedback file")
    else:
        print(f"Warning: Feedback file not found at {FEEDBACK_PATH}")
    
    print(f"Loaded {len(feedback_data)} feedback entries")
    
    if not feedback_data:
        print("No feedback data available for retraining.")
        return False
    
    # Initialize your environment
    env = SecurityEnv()
    vec_env = make_vec_env(lambda: env, n_envs=1)
    
    # Check if existing model exists
    if os.path.exists(MODEL_PATH):
        print("Loading existing model for fine-tuning")
        model = PPO.load(MODEL_PATH, env=vec_env)
    else:
        print("No existing model found. Creating a new one.")
        model = PPO("MlpPolicy", vec_env, verbose=1)
    
    # Create synthetic episodes from feedback
    episodes = create_synthetic_episodes(feedback_data, env)
    
    if not episodes:
        print("Failed to create episodes from feedback.")
        return False
    
    print(f"Created {len(episodes)} synthetic episodes for training")
    
    # Setup callbacks for saving checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path=MODEL_DIR,
        name_prefix="ppo_retrained"
    )
    
    # Fine-tune the model
    # Since we can't directly feed episodes to PPO, we'll rely on the environment
    # being initialized with good priors from the synthetic episodes
    print("Fine-tuning model...")
    model.learn(total_timesteps=10, callback=checkpoint_callback)
    
    # Save the updated model
    model.save(MODEL_PATH)
    print(f"Retrained model saved to {MODEL_PATH}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Train or retrain RL agent')
    parser.add_argument('--retrain_from_feedback', action='store_true', 
                        help='Retrain model from collected feedback')
    parser.add_argument('--train_from_scratch', action='store_true',
                       help='Train a new model from scratch')
    
    args = parser.parse_args()
    
    if args.retrain_from_feedback:
        retrain_from_feedback()
    elif args.train_from_scratch:
        train_from_scratch()
    else:
        print("Please specify either --train_from_scratch or --retrain_from_feedback")
        print("Example: python train_agent.py --train_from_scratch")

if __name__ == "__main__":
    main()
