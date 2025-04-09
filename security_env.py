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

    # Define feature types and their ranges
    FEATURE_TYPES = {
        'Level of familiarity with cybersecurity practices': 'categorical',
        'Frequency of Password Changes': 'categorical',
        'Difficulty Level': 'categorical',
        'Effort Required': 'categorical',
        'Perceived Importance': 'categorical',
        'Frequency of MFA prompts': 'categorical',
        'Difficulty Level MFA': 'categorical',
        'Effort Required MFA': 'categorical',
        'Perceived Importance of MFA': 'categorical',
        'Frequency of Security Warnings': 'categorical',
        'Difficulty Level Security Warnings': 'categorical',
        'Effort Required Security Warnings': 'categorical',
        'Perceived Importance of Security Warnings': 'categorical',
        'Which types of MFA do you encounter most often? (Select all that apply)_Authentication app (e.g., Google Authenticator, Microsoft Authenticator)': 'binary',
        'Which types of MFA do you encounter most often? (Select all that apply)_Biometric verification (fingerprint, facial recognition)': 'binary',
        'Which types of MFA do you encounter most often? (Select all that apply)_I do not use MFA': 'binary',
        'Which types of MFA do you encounter most often? (Select all that apply)_One-time passwords (OTP) via SMS or email': 'binary',
        'Which types of MFA do you encounter most often? (Select all that apply)_Security key or hardware token': 'binary',
        'Which types of security warnings do you encounter most often? (Select all that apply)_Antivirus/Malware Notifications': 'binary',
        'Which types of security warnings do you encounter most often? (Select all that apply)_I do not encounter any warnings/notifications': 'binary',
        'Which types of security warnings do you encounter most often? (Select all that apply)_Phishing Warnings': 'binary',
        'Which types of security warnings do you encounter most often? (Select all that apply)_System Update Alerts': 'binary',
        'Which types of security warnings do you encounter most often? (Select all that apply)_Unauthorized Access Attempts': 'binary'
    }

    # Define feature ranges based on the dataset
    FEATURE_RANGES = {
        'Level of familiarity with cybersecurity practices': [0, 1, 2],  # Beginner, Intermediate, Advanced
        'Frequency of Password Changes': [1, 2, 3, 4, 5],  # Daily to Annually
        'Difficulty Level': [1, 2, 3, 4, 5],  # Very easy to Very difficult
        'Effort Required': [1, 2, 3, 4, 5],  # No effort to Very high effort
        'Perceived Importance': [1, 2, 3, 4, 5],  # Not important to Very important
        'Frequency of MFA prompts': [1, 2, 3, 4, 5],  # Daily to Annually
        'Difficulty Level MFA': [1, 2, 3, 4, 5],  # Very easy to Very difficult
        'Effort Required MFA': [1, 2, 3, 4, 5],  # No effort to Very high effort
        'Perceived Importance of MFA': [1, 2, 3, 4, 5],  # Not important to Very important
        'Frequency of Security Warnings': [1, 2, 3, 4, 5],  # Daily to Annually
        'Difficulty Level Security Warnings': [1, 2, 3, 4, 5],  # Very easy to Very difficult
        'Effort Required Security Warnings': [1, 2, 3, 4, 5],  # No effort to Very high effort
        'Perceived Importance of Security Warnings': [1, 2, 3, 4, 5],  # Not important to Very important
        'Which types of MFA do you encounter most often? (Select all that apply)_Authentication app (e.g., Google Authenticator, Microsoft Authenticator)': [0, 1],
        'Which types of MFA do you encounter most often? (Select all that apply)_Biometric verification (fingerprint, facial recognition)': [0, 1],
        'Which types of MFA do you encounter most often? (Select all that apply)_I do not use MFA': [0, 1],
        'Which types of MFA do you encounter most often? (Select all that apply)_One-time passwords (OTP) via SMS or email': [0, 1],
        'Which types of MFA do you encounter most often? (Select all that apply)_Security key or hardware token': [0, 1],
        'Which types of security warnings do you encounter most often? (Select all that apply)_Antivirus/Malware Notifications': [0, 1],
        'Which types of security warnings do you encounter most often? (Select all that apply)_I do not encounter any warnings/notifications': [0, 1],
        'Which types of security warnings do you encounter most often? (Select all that apply)_Phishing Warnings': [0, 1],
        'Which types of security warnings do you encounter most often? (Select all that apply)_System Update Alerts': [0, 1],
        'Which types of security warnings do you encounter most often? (Select all that apply)_Unauthorized Access Attempts': [0, 1]
    }

    def __init__(self,
                 rf_model_path: str = "fatigue_model.joblib",
                 alpha: float = 0.3,
                 beta: float = 0.7,
                 s_min: float = 8.0,
                 max_steps: int = 100,
                 render_mode: str = None):
        super(SecurityEnv, self).__init__()

        # Initialize feature names and types
        self.feature_names = list(self.FEATURE_TYPES.keys())
        self.feature_types = self.FEATURE_TYPES
        self.feature_ranges = self.FEATURE_RANGES
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

        # Define the action space: one discrete value per feature
        action_space = []
        for feature_name in self.feature_names:
            if self.feature_types[feature_name] == 'binary':
                action_space.append(2)  # 0 or 1
            else:
                # Use the length of the feature range for categorical features
                action_space.append(len(self.feature_ranges[feature_name]))
        self.action_space = spaces.MultiDiscrete(action_space)

        # Define the observation space: configuration vector plus predicted fatigue and security score
        low_features = [0] * self.num_features
        high_features = [len(self.feature_ranges[name]) - 1 for name in self.feature_names]
        low_extra = [0.0, 0.0]  # predicted fatigue and security score lower bounds
        high_extra = [100.0, 20.0]  # upper bounds
        self.observation_space = spaces.Box(
            np.array(low_features + low_extra, dtype=np.float32),
            np.array(high_features + high_extra, dtype=np.float32),
            dtype=np.float32
        )

        # Initialize state
        self.state = None
        self.render_mode = render_mode

    def _compute_security_score(self, config: np.ndarray) -> float:
        """Compute security score using weighted sum of specific features."""
        weights = {
            # MFA weights
            "Which types of MFA do you encounter most often? (Select all that apply)_Authentication app (e.g., Google Authenticator, Microsoft Authenticator)": 2,
            "Which types of MFA do you encounter most often? (Select all that apply)_Biometric verification (fingerprint, facial recognition)": 3,
            "Which types of MFA do you encounter most often? (Select all that apply)_I do not use MFA": 0,
            "Which types of MFA do you encounter most often? (Select all that apply)_One-time passwords (OTP) via SMS or email": 2,
            "Which types of MFA do you encounter most often? (Select all that apply)_Security key or hardware token": 4,
            # Security Warnings weights
            "Which types of security warnings do you encounter most often? (Select all that apply)_Antivirus/Malware Notifications": 2,
            "Which types of security warnings do you encounter most often? (Select all that apply)_I do not encounter any warnings/notifications": 0,
            "Which types of security warnings do you encounter most often? (Select all that apply)_Phishing Warnings": 2,
            "Which types of security warnings do you encounter most often? (Select all that apply)_System Update Alerts": 2,
            "Which types of security warnings do you encounter most often? (Select all that apply)_Unauthorized Access Attempts": 3
        }
        
        score = 0.0
        for i, feature_name in enumerate(self.feature_names):
            if feature_name in weights:
                value = self._map_action_to_feature_range(config[i], feature_name)
                score += value * weights[feature_name]
        return score

    def _map_action_to_feature_range(self, action: int, feature_name: str) -> float:
        """Map discrete action to actual feature value based on feature type."""
        feature_type = self.feature_types.get(feature_name, 'categorical')
        feature_range = self.feature_ranges.get(feature_name, [0, 1, 2, 3, 4])

        # Explicit rounding and clamping
        index = int(round(action))  # ensures float like 0.0 is safe
        index = min(index, len(feature_range) - 1)
        index = max(index, 0)

        if feature_type == 'binary':
            return float(index)  # just 0 or 1
        else:
            return float(feature_range[index])


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
            action: Array of discrete values for each feature
            
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
        config = np.zeros(self.num_features, dtype=np.int64)
        for i, feature_name in enumerate(self.feature_names):
            if self.feature_types[feature_name] == 'binary':
                config[i] = self.np_random.integers(0, 2)
            else:
                # Use the length of the feature range for categorical features
                config[i] = self.np_random.integers(0, len(self.feature_ranges[feature_name]))
        
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

    # In security_env.py
    def reset_with_user_config(self, user_config: np.ndarray):
        """
        Reset the environment to a user-supplied configuration instead of random.
        user_config: a numpy array of length self.num_features,
                    each entry is an integer index for that feature.
        """
        self.current_step = 0

        # Validate that user_config length matches the environment's features
        if len(user_config) != self.num_features:
            raise ValueError("User config must have length {}".format(self.num_features))

        # Clip or adjust user_config so it doesn't exceed the valid range
        config = np.zeros(self.num_features, dtype=np.int64)
        for i, feature_name in enumerate(self.feature_names):
            max_index = len(self.feature_ranges[feature_name]) - 1
            config[i] = min(user_config[i], max_index)

        s_total = self._compute_security_score(config)
        afs = self._predict_fatigue_score(config)

        self.state = np.array(list(config) + [afs, s_total], dtype=np.float32)

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

