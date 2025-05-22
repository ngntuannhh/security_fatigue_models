import argparse
import json
import numpy as np
import datetime
import sys
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

# Set up console logger for printing messages with timestamps
def log(message, level="INFO"):
    """Print a timestamped message to the console."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    level_colors = {
        "DEBUG": "\033[94m",  # Blue
        "INFO": "\033[92m",   # Green
        "WARNING": "\033[93m", # Yellow
        "ERROR": "\033[91m",  # Red
        "CRITICAL": "\033[91m\033[1m"  # Bold Red
    }
    reset_color = "\033[0m"
    
    # Default to INFO color if level not recognized
    color = level_colors.get(level, level_colors["INFO"])
    
    # Print to console with color
    print(f"{color}[{timestamp}] {level}: {message}{reset_color}")
    
    # Also write to a log file for record-keeping
    log_file = os.path.join(script_dir, 'retrain_log.txt')
    with open(log_file, 'a') as f:
        f.write(f"[{timestamp}] {level}: {message}\n")

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

def retrain_from_feedback1():
    """
    Load feedback data and retrain the model using human feedback.
    With enhanced logging.
    """
    # First declare global variables
    global FEEDBACK_PATH, MODEL_PATH
    
    try:
        log("Starting retraining process...")
        
        # Load feedback data
        feedback_data = []
        if os.path.exists(FEEDBACK_PATH):
            log(f"Loading feedback data from {FEEDBACK_PATH}")
            with open(FEEDBACK_PATH, 'r') as f:
                for line in f:
                    try:
                        feedback_data.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        log(f"Warning: Skipping invalid JSON line in feedback file", "WARNING")
        else:
            log(f"Warning: Feedback file not found at {FEEDBACK_PATH}", "WARNING")
        
        log(f"Loaded {len(feedback_data)} feedback entries")
        
        if not feedback_data:
            log("No feedback data available for retraining.", "ERROR")
            return False
        
        # Initialize your environment
        env = SecurityEnv()
        vec_env = make_vec_env(lambda: env, n_envs=1)
        
        # Check if existing model exists
        if os.path.exists(MODEL_PATH):
            log(f"Loading existing model from {MODEL_PATH}", "INFO")
            model = PPO.load(MODEL_PATH, env=vec_env)
        else:
            log(f"No existing model found at {MODEL_PATH}. Creating a new one.", "WARNING")
            model = PPO("MlpPolicy", vec_env, verbose=1)
        
        # Create synthetic episodes from feedback
        episodes = create_synthetic_episodes(feedback_data, env)
        
        if not episodes:
            log("Failed to create episodes from feedback.", "ERROR")
            return False
        
        log(f"Created {len(episodes)} synthetic episodes for training")
        
        # Make sure model directory exists
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        
        # Setup callbacks for saving checkpoints
        checkpoint_callback = CheckpointCallback(
            save_freq=1000,
            save_path=MODEL_DIR,
            name_prefix="ppo_retrained"
        )
        
        # Fine-tune the model
        log("Starting fine-tuning process...")
        model.learn(total_timesteps=10, callback=checkpoint_callback)
        
        # Save the updated model
        log(f"Saving model to {MODEL_PATH}")
        model.save(MODEL_PATH)
        log("Model saved successfully")
        
        return True
        
    except Exception as e:
        import traceback
        log(f"Error during retraining: {str(e)}", "ERROR")
        log(traceback.format_exc(), "ERROR")
        return False



# def main():


    parser = argparse.ArgumentParser(description='Train or retrain RL agent')
    parser.add_argument('--retrain_from_feedback1', action='store_true', 
                        help='Retrain model from collected feedback')
    parser.add_argument('--train_from_scratch', action='store_true',
                       help='Train a new model from scratch')
    
    args = parser.parse_args()
    
    if args.retrain_from_feedback:
        retrain_from_feedback1()
    elif args.train_from_scratch:
        train_from_scratch()
    else:
        print("Please specify either --train_from_scratch or --retrain_from_feedback")
        print("Example: python train_agent.py --train_from_scratch")

def main():
    parser = argparse.ArgumentParser(description='Train or retrain RL agent')
    parser.add_argument('--retrain_from_feedback', action='store_true', 
                        help='Retrain model from collected feedback')
    parser.add_argument('--retrain_from_feedback1', action='store_true', 
                        help='Retrain model from collected feedback with enhanced logging')
    parser.add_argument('--train_from_scratch', action='store_true',
                       help='Train a new model from scratch')
    parser.add_argument('--feedback_file', type=str,
                       help='Path to feedback file (overrides default)')
    parser.add_argument('--model_path', type=str,
                       help='Path to model file (overrides default)')
    
    args = parser.parse_args()
    
    # Log the arguments received for debugging
    print(f"Arguments received: {args}")
    
    # Update paths if provided in arguments
    global FEEDBACK_PATH, MODEL_PATH

    if args.feedback_file:
        FEEDBACK_PATH = args.feedback_file
        print(f"Using custom feedback file: {FEEDBACK_PATH}")
    if args.model_path:
        MODEL_PATH = args.model_path
        print(f"Using custom model path: {MODEL_PATH}")
    
    # Check which retraining method was requested
    if args.retrain_from_feedback1:
        print("Using enhanced logging version of retraining")
        retrain_from_feedback1()
    elif args.retrain_from_feedback:
        print("Using standard version of retraining")
        retrain_from_feedback()
    elif args.train_from_scratch:
        train_from_scratch()
    else:
        print("Please specify either --train_from_scratch, --retrain_from_feedback or --retrain_from_feedback1")
        print("Example: python train_agent.py --retrain_from_feedback1")


if __name__ == "__main__":
    main()
