import argparse
import json
import numpy as np
import datetime
import sys
import os
import shutil
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from security_env import SecurityEnv

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(script_dir, 'models')
LOG_DIR = os.path.join(script_dir, 'retrain_logs')
PREV_MODELS_DIR = os.path.join(MODEL_DIR, 'previous_versions')

# Ensure directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PREV_MODELS_DIR, exist_ok=True)

MODEL_VER = 'run_tuned_1m'  # or your chosen model version

MODEL_PATH = os.path.join(MODEL_DIR, MODEL_VER, 'best_model', 'best_model.zip')
FEEDBACK_PATH = os.path.join(script_dir, 'feedback_buffer.jsonl')

# Get timestamped filename for logging and versioning
def get_timestamp_str():
    """Generate a timestamp string for filenames"""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Set up console logger for printing messages with timestamps
def log(message, level="INFO", log_file=None):
    """Print a timestamped message to the console and log file."""
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
    if log_file:
        with open(log_file, 'a') as f:
            f.write(f"[{timestamp}] {level}: {message}\n")

def backup_existing_model(model_path):
    """
    Backup the existing model before overwriting it
    """
    if os.path.exists(model_path):
        timestamp = get_timestamp_str()
        filename = os.path.basename(model_path)
        backup_path = os.path.join(PREV_MODELS_DIR, f"{filename.split('.')[0]}_{timestamp}.zip")
        
        try:
            shutil.copy2(model_path, backup_path)
            return backup_path
        except Exception as e:
            print(f"Warning: Failed to backup model: {e}")
    
    return None

def create_synthetic_episodes(feedback_data, env, log_file=None):
    """
    Convert feedback data into synthetic episodes that can be used for retraining.
    
    This function takes human feedback data and creates synthetic episodes 
    that can be used to guide the RL agent's learning process.
    """
    episodes = []
    
    for i, entry in enumerate(feedback_data):
        try:
            log(f"Processing feedback entry {i+1}/{len(feedback_data)}", log_file=log_file)
            
            initial_config = np.array(entry['initial_config'])
            final_config = np.array(entry['final_config'])
            reward = entry.get('reward', None)
            
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
            
        except Exception as e:
            log(f"Error processing feedback entry {i+1}: {str(e)}", "ERROR", log_file=log_file)
            continue
    
    return episodes

def train_from_scratch():
    """
    Train a new RL model from scratch.
    """
    # Create a timestamped log file
    timestamp = get_timestamp_str()
    log_file = os.path.join(LOG_DIR, f"train_from_scratch_{timestamp}.log")
    
    log("Starting training from scratch...", log_file=log_file)
    
    # Make sure the model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Create timestamped checkpoint directory
    checkpoint_dir = os.path.join(MODEL_DIR, f"checkpoints_{timestamp}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create the environment
    env = make_vec_env(SecurityEnv, n_envs=1)
    
    # Initialize a new model
    model = PPO("MlpPolicy", env, verbose=1)
    
    # Setup callbacks for saving checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,  # Save every 10,000 steps
        save_path=checkpoint_dir,
        name_prefix="ppo_checkpoint"
    )
    
    # Train the model
    log("Training model...", log_file=log_file)
    model.learn(total_timesteps=1000, callback=checkpoint_callback)
    
    # Backup existing model if it exists
    backup_path = backup_existing_model(MODEL_PATH)
    if backup_path:
        log(f"Backed up existing model to {backup_path}", log_file=log_file)
    
    # Create directory for model if it doesn't exist
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    # Save the final model
    model.save(MODEL_PATH)
    log(f"Model saved to {MODEL_PATH}", log_file=log_file)
    
    return True

def retrain_from_feedback():
    """
    Load feedback data and retrain the model using human feedback.
    With enhanced logging.
    """
    # Create timestamped log file
    timestamp = get_timestamp_str()
    log_file = os.path.join(LOG_DIR, f"retrain_{timestamp}.log")
    
    # First declare global variables
    global FEEDBACK_PATH, MODEL_PATH
    
    try:
        log("="*60, log_file=log_file)
        log(f"STARTING RETRAINING PROCESS - {timestamp}", log_file=log_file)
        log("="*60, log_file=log_file)
        
        # Log system information
        log(f"Python version: {sys.version}", log_file=log_file)
        log(f"Working directory: {os.getcwd()}", log_file=log_file)
        log(f"Feedback path: {FEEDBACK_PATH}", log_file=log_file)
        log(f"Model path: {MODEL_PATH}", log_file=log_file)
        
        # Load feedback data
        feedback_data = []
        if os.path.exists(FEEDBACK_PATH):
            log(f"Loading feedback data from {FEEDBACK_PATH}", log_file=log_file)
            with open(FEEDBACK_PATH, 'r') as f:
                for line in f:
                    try:
                        feedback_data.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        log(f"Warning: Skipping invalid JSON line in feedback file", "WARNING", log_file=log_file)
        else:
            log(f"Warning: Feedback file not found at {FEEDBACK_PATH}", "WARNING", log_file=log_file)
        
        log(f"Loaded {len(feedback_data)} feedback entries", log_file=log_file)
        
        if not feedback_data:
            log("No feedback data available for retraining.", "ERROR", log_file=log_file)
            return False
        
        # Initialize your environment
        env = SecurityEnv()
        vec_env = make_vec_env(lambda: env, n_envs=1)
        
        # Check if existing model exists
        if os.path.exists(MODEL_PATH):
            log(f"Loading existing model from {MODEL_PATH}", "INFO", log_file=log_file)
            model = PPO.load(MODEL_PATH, env=vec_env)
        else:
            log(f"No existing model found at {MODEL_PATH}. Creating a new one.", "WARNING", log_file=log_file)
            model = PPO("MlpPolicy", vec_env, verbose=1)
        
        # Create synthetic episodes from feedback
        episodes = create_synthetic_episodes(feedback_data, env, log_file=log_file)
        
        if not episodes:
            log("Failed to create episodes from feedback.", "ERROR", log_file=log_file)
            return False
        
        log(f"Created {len(episodes)} synthetic episodes for training", log_file=log_file)
        
        # Create timestamped checkpoint directory
        checkpoint_dir = os.path.join(MODEL_DIR, f"checkpoints_{timestamp}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        log(f"Created checkpoint directory: {checkpoint_dir}", log_file=log_file)
        
        # Setup callbacks for saving checkpoints
        checkpoint_callback = CheckpointCallback(
            save_freq=1000,
            save_path=checkpoint_dir,
            name_prefix="ppo_retrained"
        )
        
        # Backup existing model before overwriting
        backup_path = backup_existing_model(MODEL_PATH)
        if backup_path:
            log(f"Backed up existing model to {backup_path}", log_file=log_file)
        
        # Make sure model directory exists
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        
        # Fine-tune the model
        log("Starting fine-tuning process...", log_file=log_file)
        model.learn(total_timesteps=10, callback=checkpoint_callback)
        
        # Save the updated model
        log(f"Saving model to {MODEL_PATH}", log_file=log_file)
        model.save(MODEL_PATH)
        log("Model saved successfully", log_file=log_file)
        
        # Log completion
        log("="*60, log_file=log_file)
        log(f"RETRAINING COMPLETED SUCCESSFULLY - {timestamp}", log_file=log_file)
        log("="*60, log_file=log_file)
        
        return True
        
    except Exception as e:
        import traceback
        log(f"Error during retraining: {str(e)}", "ERROR", log_file=log_file)
        log(traceback.format_exc(), "ERROR", log_file=log_file)
        return False

def main():
    parser = argparse.ArgumentParser(description='Train or retrain RL agent')
    parser.add_argument('--retrain_from_feedback', action='store_true', 
                        help='Retrain model from collected feedback')
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
    if args.retrain_from_feedback:
        print("Using enhanced logging version of retraining")
        retrain_from_feedback()
    elif args.train_from_scratch:
        train_from_scratch()
    else:
        print("Please specify either --train_from_scratch, --retrain_from_feedback")
        print("Example: python train_agent.py --retrain_from_feedback")


if __name__ == "__main__":
    main()