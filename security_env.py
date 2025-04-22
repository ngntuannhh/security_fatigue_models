import gymnasium as gym
from gymnasium import spaces
import numpy as np
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
        'Familarity': 'categorical',

        'Frequency of Password Changes': 'categorical',
        'Difficulty Level Password': 'categorical',
        'Effort Required Password': 'categorical',
        'Perceived Importance Password': 'categorical',
        'Password Uniqueness': 'categorical',

        'Frequency of MFA prompts': 'categorical',
        'Difficulty Level MFA': 'categorical',
        'Effort Required MFA': 'categorical',
        'Perceived Importance of MFA': 'categorical',

        'Frequency of Security Warnings': 'categorical',
        'Difficulty Level Security Warnings': 'categorical',
        'Effort Required Security Warnings': 'categorical',
        'Perceived Importance of Security Warnings': 'categorical',
        'Warnings Response Behaviour': 'categorical',

        'Hardware security key (FIDO2 token) or cryptographic device': 'binary',
        'On-device prompt or biometric': 'binary',
        'OTP via authenticator app': 'binary',
        'OTP via SMS/email': 'binary',
        'Secondary email/phone or security questions': 'binary',
        'No MFA enabled': 'binary'
    }

    # Define feature ranges based on the dataset
    FEATURE_RANGES = {
        #Familiarity with cybersecurity practices
        'Familarity': [0, 1, 2],  # Beginner, Intermediate, Advanced
        #Password
        'Frequency of Password Changes': [1, 2, 3, 4, 5],  # Annually to Daily
        'Difficulty Level Password': [1, 2, 3, 4, 5],  # Very easy to Very difficult
        'Effort Required Password': [1, 2, 3, 4, 5],  # No effort to Very high effort
        'Perceived Importance Password': [1, 2, 3, 4, 5],  # Not important to Very important
        'Password Uniqueness': [0.2, 0.5, 0.8, 1.0],  # Not unique to Very unique
        #MFA
        'Frequency of MFA prompts': [1, 2, 3, 4, 5],  # Annually to Daily
        'Difficulty Level MFA': [1, 2, 3, 4, 5],  # Very easy to Very difficult
        'Effort Required MFA': [1, 2, 3, 4, 5],  # No effort to Very high effort
        'Perceived Importance of MFA': [1, 2, 3, 4, 5],  # Not important to Very important
        #Warnings
        'Frequency of Security Warnings': [1, 2, 3, 4, 5],  # Annually to Daily
        'Difficulty Level Security Warnings': [1, 2, 3, 4, 5],  # Very easy to Very difficult
        'Effort Required Security Warnings': [1, 2, 3, 4, 5],  # No effort to Very high effort
        'Perceived Importance of Security Warnings': [1, 2, 3, 4, 5],  # Not important to Very important
        'Warnings Response Behaviour': [20, 40, 60, 80, 100],
        #MFA Types & warnings types (binary)
        'Hardware security key (FIDO2 token) or cryptographic device': [0, 1],
        'On-device prompt or biometric': [0, 1],
        'OTP via authenticator app': [0, 1],
        'OTP via SMS/email': [0, 1],
        'Secondary email/phone or security questions': [0, 1],
        'No MFA enabled': [0, 1]
    }

    def __init__(self,
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

         # Debug: print feature names
        print("Feature names:", self.feature_names)
    
        # Debug: check for missing keys in FEATURE_RANGES
        missing_keys = [name for name in self.feature_names if name not in self.FEATURE_RANGES]
        if missing_keys:
            print("WARNING: These feature names are missing from FEATURE_RANGES:", missing_keys)

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
        high_extra = [100.0, 100.0]  # upper bounds
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
        """Formula = S_pass * W_pass + S_mfa * W_mfa + S_war * W_war"""
        # Define weights for each component
        W_pass = 0.3
        W_mfa = 0.5
        W_war = 0.2
        
        # Get indices of key features for direct access
        difficulty_password_idx = self.feature_names.index('Difficulty Level Password')
        password_uniqueness_idx = self.feature_names.index('Password Uniqueness')
        mfa_frequency_idx = self.feature_names.index('Frequency of MFA prompts')
        warnings_response_idx = self.feature_names.index('Warnings Response Behaviour')
        
        # 1. Calculate Password Security Score
        password_difficulty_map = {
            1: 20,  # Very easy -> 20
            2: 40,  # Easy -> 40
            3: 60,  # Moderate -> 60
            4: 80,  # Difficult -> 80
            5: 100  # Very difficult -> 100
        }
        
        # Get mapped difficulty and uniqueness values
        difficulty_value = self._map_action_to_feature_range(config[difficulty_password_idx], 'Difficulty Level Password')
        uniqueness_value = self._map_action_to_feature_range(config[password_uniqueness_idx], 'Password Uniqueness')
        
        # Calculate S_pass
        S_pass = password_difficulty_map.get(difficulty_value, 0) * uniqueness_value
        
        # 2. Calculate MFA Security Score
        mfa_frequency_map = {
            1: 0.2,    # annually -> 100%
            2: 0.5,  # Weekly -> 90%
            3: 0.75,  # Monthly -> 75%
            4: 0.9,  # Quarterly -> 50%
            5: 1   # daily -> 20%
        }
        
        # Get MFA frequency factor
        frequency_value = self._map_action_to_feature_range(config[mfa_frequency_idx], 'Frequency of MFA prompts')
        frequency_factor = mfa_frequency_map.get(frequency_value, 0.5)
        
        # MFA type weights
        mfa_weights = {
            'Hardware security key (FIDO2 token) or cryptographic device': 100,
            'On-device prompt or biometric': 90,
            'OTP via authenticator app': 80,
            'OTP via SMS/email': 70,
            'Secondary email/phone or security questions': 40,
            'No MFA enabled': 0
        }
        
        # Calculate S_MFA
        S_MFA = 0
        max_mfa_score = 0
        for mfa_type, weight in mfa_weights.items():
            if mfa_type in self.feature_names:
                idx = self.feature_names.index(mfa_type)
                is_enabled = self._map_action_to_feature_range(config[idx], mfa_type)
                if is_enabled == 1 and weight > max_mfa_score:
                    max_mfa_score = weight
        
        # Apply frequency factor to MFA score
        S_MFA = max_mfa_score * frequency_factor
        
        # 3. Calculate Warning Response Security Score
        S_war = self._map_action_to_feature_range(config[warnings_response_idx], 'Warnings Response Behaviour')
        
        # Combined security score
        security_score = W_pass * S_pass + W_mfa * S_MFA + W_war * S_war
        
        return security_score

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


    def _compute_fatigue_score(self, config: np.ndarray) -> float:
        """
        Calculate fatigue score using the formula:
        Fatigue Score = (Frequency × Difficulty × Effort) / (Importance × (1 + 0.5 × (Familiarity / 2)))
        
        This is calculated for each security measure (password, MFA, warnings) and then combined.
        """
        try:
            # Get feature values from configuration
            feature_values = {}
            for i, feature_name in enumerate(self.feature_names):
                feature_values[feature_name] = self._map_action_to_feature_range(config[i], feature_name)
            
            # Get familiarity value (common to all calculations)
            familiarity = feature_values.get('Familarity', 1)  # Default to intermediate (1) if not found
            familiarity_factor = 1 + 0.5 * (familiarity / 2)
            
            # Calculate Password Fatigue
            password_fatigue = 0
            try:
                password_freq = feature_values['Frequency of Password Changes']
                password_diff = feature_values['Difficulty Level Password']
                password_effort = feature_values['Effort Required Password']
                password_importance = feature_values['Perceived Importance Password']
                
                if password_importance > 0:  # Avoid division by zero
                    password_fatigue = (password_freq * password_diff * password_effort) / (password_importance)
            except Exception as e:
                print(f"Error calculating password fatigue: {e}")
            
            # Calculate MFA Fatigue
            mfa_fatigue = 0
            try:
                # Only calculate MFA fatigue if MFA is enabled
                mfa_enabled = (
                    feature_values.get('Hardware security key (FIDO2 token) or cryptographic device', 0) or
                    feature_values.get('On-device prompt or biometric', 0) or
                    feature_values.get('OTP via authenticator app', 0) or
                    feature_values.get('OTP via SMS/email', 0) or
                    feature_values.get('Secondary email/phone or security questions', 0)
                )
                
                if mfa_enabled and feature_values.get('No MFA enabled', 0) == 0:
                    mfa_freq = feature_values.get('Frequency of MFA prompts', 3)
                    mfa_diff = feature_values.get('Difficulty Level MFA', 3) 
                    mfa_effort = feature_values.get('Effort Required MFA', 3)
                    mfa_importance = feature_values.get('Perceived Importance of MFA', 3)
                    
                    if mfa_importance > 0:  # Avoid division by zero
                        mfa_fatigue = (mfa_freq * mfa_diff * mfa_effort) / (mfa_importance)
            except Exception as e:
                print(f"Error calculating MFA fatigue: {e}")
            
            # Calculate Warnings Fatigue
            warnings_fatigue = 0
            try:
                warnings_freq = feature_values.get('Frequency of Security Warnings', 3)
                warnings_diff = feature_values.get('Difficulty Level Security Warnings', 3)
                warnings_effort = feature_values.get('Effort Required Security Warnings', 3)
                warnings_importance = feature_values.get('Perceived Importance of Security Warnings', 3)
                
                if warnings_importance > 0:  # Avoid division by zero
                    warnings_fatigue = (warnings_freq * warnings_diff * warnings_effort) / (warnings_importance)
            except Exception as e:
                print(f"Error calculating warnings fatigue: {e}")
            
            # Combine fatigue scores using simple sum
            total_fatigue = (password_fatigue + mfa_fatigue + warnings_fatigue)/3 * familiarity_factor
            
            # Normalize to a 0-100 scal
            max_possible_fatigue = 187.5
            normalized_fatigue = min(100, (total_fatigue / max_possible_fatigue) * 100)
            
            return normalized_fatigue
            
        except Exception as e:
            print(f"Error computing fatigue score: {e}")
            return 50.0  # Default moderate fatigue if calculation fails

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
        
        # Make a copy of the action to avoid modifying the original
        adjusted_action = action.copy()
    
        # Find indices of all MFA-related features
        mfa_type_indices = []
        no_mfa_index = None
    
        for i, name in enumerate(self.feature_names):
            if name == 'No MFA enabled':
                no_mfa_index = i
            elif any(x in name for x in ['FIDO2', 'biometric', 'OTP', 'SMS', 'email', 'security questions']):
                mfa_type_indices.append((i, name))
    
        # MFA Constraint Logic:
        if no_mfa_index is not None:
            if adjusted_action[no_mfa_index] == 1:
                # If "No MFA enabled" is selected, disable all other MFA options
                for idx, _ in mfa_type_indices:
                    adjusted_action[idx] = 0
            else:
                # If "No MFA enabled" is not selected, ensure exactly one MFA option is enabled
                
                # Count how many MFA types are enabled
                enabled_mfa_count = sum(adjusted_action[idx] == 1 for idx, _ in mfa_type_indices)
                
                if enabled_mfa_count == 0:
                    # If none are enabled, enable the first one by default
                    if mfa_type_indices:
                        adjusted_action[mfa_type_indices[0][0]] = 1
                elif enabled_mfa_count > 1:
                    # If multiple are enabled, keep only the highest priority one
                    # Find the first enabled MFA type (using the order defined in your feature names)
                    enabled_indices = [(idx, name) for idx, name in mfa_type_indices if adjusted_action[idx] == 1]
                    
                    # Define MFA type priority (highest to lowest)
                    mfa_priority = [
                        'Hardware security key (FIDO2 token) or cryptographic device',
                        'On-device prompt or biometric',
                        'OTP via authenticator app',
                        'OTP via SMS/email',
                        'Secondary email/phone or security questions'
                    ]
                    
                    # Find the highest priority enabled MFA type
                    highest_priority_index = None
                    highest_priority_rank = float('inf')
                    
                    for idx, name in enabled_indices:
                        if name in mfa_priority:
                            rank = mfa_priority.index(name)
                            if rank < highest_priority_rank:
                                highest_priority_rank = rank
                                highest_priority_index = idx
                    
                    # Keep only the highest priority one enabled
                    if highest_priority_index is not None:
                        for idx, _ in mfa_type_indices:
                            if idx != highest_priority_index:
                                adjusted_action[idx] = 0

        # Update configuration and compute scores
        config = adjusted_action
        s_total = self._compute_security_score(config)
        afs = self._compute_fatigue_score(config)
        
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
        afs = self._compute_fatigue_score(config)
        
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
        afs = self._compute_fatigue_score(config)

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

