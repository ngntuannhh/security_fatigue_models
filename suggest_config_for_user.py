import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from security_env import SecurityEnv

model_version = 'run_tuned_1m' # or your chosen model version

# Get the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, 'models')
path = os.path.join(model_dir, model_version, 'best_model', 'best_model.zip')

# path = os.path.join(script_dir, 'models', 'run_default_20250421_212017', 'best_model', 'best_model.zip')

# path = r"C:\Users\Tuan Anh HSLU\OneDrive - Hochschule Luzern\Desktop\HSLU22\Bachelor Thesis\ML Models\models\run_default_20250421_212017\best_model\best_model.zip"

def suggest_config_for_user(user_config: np.ndarray,
                            model_path: str = path,
                            n_steps: int = 100):
    """
    1. Loads the pre-trained RL model (PPO).
    2. Resets the SecurityEnv to the user's config. 
    3. Steps through the environment using the model's policy.
    4. Returns the final proposed configuration plus any info.
    """
    # Create the environment
    env = SecurityEnv(
        alpha=0.7,  # or your chosen alpha
        beta=0.3,   # or your chosen beta
        s_min=5.0
    )

    # Load your trained PPO agent
    model = PPO.load(model_path, env=env)
    # Reset environment to the user config
    obs, info = env.reset_with_user_config(user_config)
    
    final_obs = None
    for step in range(n_steps):
        action, _states = model.predict(obs, deterministic=False)
        obs, reward, done, truncated, info = env.step(action)
        final_obs = obs  # keep the latest state
        if done or truncated:
            break

    if final_obs is None:
        raise RuntimeError("Model did not perform any step; final state is undefined.")
    
    # The 'obs' now contains the final state after n_steps, i.e. the final config + [afs, security]
    final_config = obs[:-2]  # everything except the last two (fatigue, security)
    predicted_fatigue = obs[-2]
    security_score = obs[-1]
    
    # Convert final_config to integers for compact representation
    final_config_int = np.round(final_config).astype(np.int64).tolist()
    
    # Create a compact response with just the essential information
    suggestion = {
        "optimized_config": final_config_int,  # The configuration as integers
        "fatigue": float(predicted_fatigue),
        "security": float(security_score),
        "steps": step + 1
    }
    
    # Optional: For debugging, you can still generate the detailed feature values
    if False:  # Set to True for debugging
        # Convert final_config to integers before passing to _get_feature_values
        final_config_int_arr = final_config.astype(np.int64)
        # Convert final_config from indexes back to meaningful feature values
        feature_values = env._get_feature_values(final_config_int_arr)
        
        print("\n🔎 Mapping Check: Index → Value")
        for i, feature in enumerate(env.feature_names):
            idx = int(round(final_config[i]))
            expected = env.feature_ranges[feature][idx]
            actual = feature_values[feature]
            print(f"{feature}: Index = {idx}, Mapped = {expected}, FeatureValue = {actual}, Match = {expected == actual}")
        
        suggestion["feature_values"] = feature_values
    
    return suggestion
