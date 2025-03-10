import gymnasium as gym
from gymnasium import spaces
import numpy as np
import joblib
import pandas as pd
from typing import Tuple, Dict, Any, List

class SecurityEnv(gym.Env):
    """
    RL environment that proposes a configuration vector to match the updated RF model.
    The RF model expects an input vector of length n, which is determined automatically.

    The state consists of:
      - The configuration vector (n features; assumed discrete values 0, 1, or 2)
      - The predicted fatigue (output from the RF model)
      - The computed security score (sum of feature values; maximum = 2*n)

    Reward:
      R = alpha * (security_score - s_min) - beta * (predicted_fatigue)
    """
    metadata = {'render_modes': ['human']}

    # Define feature types and ranges as class variables
    FEATURE_TYPES = {
        'Level of familiarity with cybersecurity practices': 'categorical',  # 0, 1, 2
        'Frequency of Password Changes': 'ordinal',  # 1-5
        'Difficulty Level': 'ordinal',  # 1-5
        'Effort Required': 'ordinal',  # 1-5
        'Perceived Importance': 'ordinal',  # 1-5
        'Frequency of MFA prompts': 'ordinal',  # 1-5
        'Difficulty Level MFA': 'ordinal',  # 1-5
        'Effort Required MFA': 'ordinal',  # 1-5
        'Perceived Importance of MFA': 'ordinal',  # 1-5
        'Frequency of Security Warnings': 'ordinal',  # 1-5
        'Difficulty Level Security Warnings': 'ordinal',  # 1-5
        'Effort Required Security Warnings': 'ordinal',  # 1-5
        'Perceived Importance of Security Warnings': 'ordinal',  # 1-5
        'Which types of MFA do you encounter most often? (Select all that apply)_Authentication app': 'binary',
        'Which types of MFA do you encounter most often? (Select all that apply)_Biometric verification': 'binary',
        'Which types of MFA do you encounter most often? (Select all that apply)_I do not use MFA': 'binary',
        'Which types of MFA do you encounter most often? (Select all that apply)_One-time passwords (OTP) via SMS or email': 'binary',
        'Which types of Security Warnings do you encounter most often? (Select all that apply)_Email notifications': 'binary',
        'Which types of Security Warnings do you encounter most often? (Select all that apply)_In-app notifications': 'binary',
        'Which types of Security Warnings do you encounter most often? (Select all that apply)_Pop-up alerts': 'binary',
        'Which types of Security Warnings do you encounter most often? (Select all that apply)_SMS notifications': 'binary'
    }

    FEATURE_RANGES = {
        'Level of familiarity with cybersecurity practices': (0, 2),
        'Frequency of Password Changes': (1, 5),
        'Difficulty Level': (1, 5),
        'Effort Required': (1, 5),
        'Perceived Importance': (1, 5),
        'Frequency of MFA prompts': (1, 5),
        'Difficulty Level MFA': (1, 5),
        'Effort Required MFA': (1, 5),
        'Perceived Importance of MFA': (1, 5),
        'Frequency of Security Warnings': (1, 5),
        'Difficulty Level Security Warnings': (1, 5),
        'Effort Required Security Warnings': (1, 5),
        'Perceived Importance of Security Warnings': (1, 5)
    }

    def __init__(self,
                 rf_model_path: str = "fatigue_model.joblib",
                 alpha: float = 0.7,
                 beta: float = 0.3,
                 s_min: float = 5.0,
                 max_steps: int = 100,
                 render_mode: str = None):
        super(SecurityEnv, self).__init__()

        # Initialize feature names and types
        self.feature_types = self.FEATURE_TYPES
        self.feature_ranges = self.FEATURE_RANGES
        self.feature_names = list(self.FEATURE_TYPES.keys())
        self.num_features = len(self.feature_names)

        # Load the trained Random Forest model
        try:
            self.fatigue_model = joblib.load(rf_model_path)
            # Verify that model features match our defined features
            model_features = self.fatigue_model.feature_names_in_
            if not all(f in self.feature_names for f in model_features):
                print("Warning: Model features don't match defined features!")
                print("Model features:", model_features)
                print("Defined features:", self.feature_names)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using default feature configuration...")

        # RL hyperparameters
        self.alpha = alpha
        self.beta = beta
        self.s_min = s_min
        self.max_steps = max_steps
        self.current_step = 0

        # Define the action space: one discrete value per feature (values 0, 1, or 2)
        self.action_space = spaces.MultiDiscrete([3] * self.num_features)

        # Define the observation space: configuration vector plus predicted fatigue and security score
        low_features = [0] * self.num_features
        high_features = [2] * self.num_features
        low_extra = [0.0, 0.0]  # predicted fatigue and security score lower bounds
        high_extra = [10.0, 2 * self.num_features]  # example upper bounds
        self.observation_space = spaces.Box(
            np.array(low_features + low_extra, dtype=np.float32),
            np.array(high_features + high_extra, dtype=np.float32),
            dtype=np.float32
        )

        # Initialize state
        self.state = None
        self.render_mode = render_mode

    def _compute_security_score(self, config: np.ndarray) -> float:
        """Compute security score as the sum of feature values."""
        return float(np.sum(config))

    def _map_action_to_feature_range(self, action: int, feature_name: str) -> float:
        """Map discrete action (0,1,2) to actual feature range based on feature type."""
        feature_type = self.feature_types.get(feature_name, 'ordinal')
        
        if feature_type == 'binary':
            return float(action >= 1)  # Convert to 0 or 1
        elif feature_type == 'categorical':
            return float(action)  # Keep as 0, 1, or 2
        else:  # ordinal (1-5)
            min_val, max_val = self.feature_ranges.get(feature_name, (1, 5))
            if action == 0:
                return min_val
            elif action == 1:
                return (min_val + max_val) / 2
            else:  # action == 2
                return max_val

    def _predict_fatigue_score(self, config: np.ndarray) -> float:
        """Predict fatigue score using the RF model."""
        try:
            # Map discrete actions to actual feature values
            feature_values = []
            for i, feature_name in enumerate(self.feature_names):
                feature_values.append(self._map_action_to_feature_range(config[i], feature_name))

            # Create DataFrame with proper feature names
            X = pd.DataFrame([feature_values], columns=self.feature_names)
            return self.fatigue_model.predict(X)[0]
        except Exception as e:
            print(f"Error predicting fatigue score: {e}")
            return 0.0  # Return default value if prediction fails

    def _get_feature_values(self, config: np.ndarray) -> Dict[str, float]:
        """Get the actual feature values for a given configuration."""
        feature_values = {}
        for i, feature_name in enumerate(self.feature_names):
            feature_values[feature_name] = self._map_action_to_feature_range(config[i], feature_name)
        return feature_values

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Array of discrete values (0, 1, or 2) for each feature
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        self.current_step += 1
        
        # Update configuration and compute scores
        config = action
        s_total = self._compute_security_score(config)
        afs = self._predict_fatigue_score(config)
        
        # Compute reward
        reward = self.alpha * (s_total - self.s_min) - self.beta * afs
        
        # Update state
        self.state = np.array(list(config) + [afs, s_total], dtype=np.float32)
        
        # Check termination conditions
        terminated = False  # Can be customized based on specific requirements
        truncated = self.current_step >= self.max_steps
        
        # Additional info
        info = {
            'security_score': s_total,
            'fatigue_score': afs,
            'step': self.current_step,
            'feature_values': self._get_feature_values(config)
        }
        
        return self.state, reward, terminated, truncated, info

    def reset(self, seed: int = None, options: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset
            
        Returns:
            tuple: (observation, info)
        """
        super().reset(seed=seed)
        
        # Reset step counter
        self.current_step = 0
        
        # Generate random initial configuration
        config = self.np_random.integers(0, 3, size=(self.num_features,))
        s_total = self._compute_security_score(config)
        afs = self._predict_fatigue_score(config)
        
        # Set initial state
        self.state = np.array(list(config) + [afs, s_total], dtype=np.float32)
        
        # Create info dictionary with feature values
        info = {
            'security_score': s_total,
            'fatigue_score': afs,
            'step': self.current_step,
            'feature_values': self._get_feature_values(config)
        }
        
        return self.state, info

    def render(self) -> None:
        """Render the environment to the screen."""
        if self.render_mode == "human":
            print(f"Step: {self.current_step}")
            print("Feature Values:")
            for i, name in enumerate(self.feature_names):
                value = self._map_action_to_feature_range(self.state[i], name)
                print(f"  {name}: {value}")
            print(f"Fatigue Score: {self.state[-2]:.2f}")
            print(f"Security Score: {self.state[-1]:.2f}")
            print("-" * 50)

    def close(self) -> None:
        """Clean up resources."""
        pass 