#!/usr/bin/env python3
"""
Bayesian Sensor Fusion Optimization
Optimizes depth image preprocessing, frame stacking, and proprioceptive feature engineering
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import warnings
import copy

try:
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
    from botorch.optim import optimize_acqf
    from gpytorch.mlls import ExactMarginalLogLikelihood
    BOTORCH_AVAILABLE = True
except ImportError:
    BOTORCH_AVAILABLE = False

class PreprocessingMethod(Enum):
    """Depth image preprocessing methods"""
    NONE = "none"
    GAUSSIAN_BLUR = "gaussian_blur"
    BILATERAL_FILTER = "bilateral_filter"
    MEDIAN_FILTER = "median_filter"
    MORPHOLOGICAL_OPENING = "morphological_opening"
    EDGE_PRESERVING = "edge_preserving"

class NormalizationMethod(Enum):
    """Depth normalization strategies"""
    MIN_MAX = "min_max"
    Z_SCORE = "z_score"
    ROBUST = "robust"  # Use percentiles instead of mean/std
    ADAPTIVE = "adaptive"  # Dynamic range adaptation
    LOG_SCALE = "log_scale"

class FrameStackingStrategy(Enum):
    """Frame stacking temporal strategies"""
    UNIFORM = "uniform"  # Equal spacing
    EXPONENTIAL = "exponential"  # More recent frames weighted higher
    FIBONACCI = "fibonacci"  # Fibonacci spacing
    ADAPTIVE = "adaptive"  # Dynamic spacing based on motion

@dataclass
class SensorFusionConfig:
    """Configuration for sensor fusion optimization"""
    # Depth preprocessing
    preprocessing_method: PreprocessingMethod
    preprocessing_strength: float  # 0.0 to 1.0
    normalization_method: NormalizationMethod
    depth_range_min: float  # meters
    depth_range_max: float  # meters
    
    # Frame stacking
    frame_stack_size: int  # 1 to 8
    temporal_spacing: float  # seconds between frames
    stacking_strategy: FrameStackingStrategy
    frame_interpolation: bool  # interpolate missing frames
    
    # Feature extraction
    spatial_pooling_size: int  # pooling kernel size
    feature_pyramid_levels: int  # 1 to 4
    attention_mechanism: bool  # use spatial attention
    
    # Proprioceptive fusion
    imu_weight: float  # 0.0 to 1.0
    odometry_weight: float  # 0.0 to 1.0
    motor_encoder_weight: float  # 0.0 to 1.0
    fusion_method: str  # "concatenate", "weighted_sum", "attention"
    
    # Temporal modeling
    use_lstm: bool
    lstm_hidden_size: int
    sequence_length: int  # frames to remember
    
    # Advanced features
    use_uncertainty_estimation: bool
    confidence_threshold: float  # 0.0 to 1.0
    failure_detection: bool

class BayesianSensorFusionOptimizer:
    """
    Bayesian optimization for sensor fusion parameters.
    
    Optimizes depth image preprocessing, frame stacking strategies,
    and proprioceptive sensor fusion for maximum navigation performance.
    """
    
    def __init__(self,
                 depth_image_shape: Tuple[int, int] = (480, 640),
                 max_frame_stack: int = 6,
                 enable_debug: bool = False):
        
        if not BOTORCH_AVAILABLE:
            raise ImportError("BoTorch is required for sensor fusion optimization. Install with: pip install botorch")
        
        self.depth_image_shape = depth_image_shape
        self.max_frame_stack = max_frame_stack
        self.enable_debug = enable_debug
        
        # Define optimization search space
        self.search_space = self._define_search_space()
        self.param_names = list(self.search_space.keys())
        self.n_params = len(self.param_names)
        
        # Convert to BoTorch bounds
        bounds_array = np.array([self.search_space[name] for name in self.param_names])
        self.bounds = torch.tensor(bounds_array.T, dtype=torch.float64)
        
        # Optimization history
        self.config_history: List[SensorFusionConfig] = []
        self.performance_history: List[float] = []
        self.parameter_history: List[torch.Tensor] = []
        
        # BoTorch components
        self.gp_model: Optional[SingleTaskGP] = None
        self.mll = None
        
        # Best configuration tracking
        self.best_config: Optional[SensorFusionConfig] = None
        self.best_performance: float = -float('inf')
        
        # Performance metrics tracking
        self.performance_components = {
            'navigation_efficiency': [],
            'collision_avoidance': [],
            'computational_cost': [],
            'robustness_score': []
        }
        
        if self.enable_debug:
            print(f"[SensorFusionOptimizer] Initialized with {self.n_params} parameters")
            print(f"[SensorFusionOptimizer] Depth image shape: {depth_image_shape}")
    
    def _define_search_space(self) -> Dict[str, Tuple[float, float]]:
        """Define the sensor fusion optimization search space"""
        
        return {
            # Depth preprocessing (0-5 maps to enum indices)
            'preprocessing_method': (0, 5),
            'preprocessing_strength': (0.1, 1.0),
            'normalization_method': (0, 4),  # 0-4 maps to enum indices
            'depth_range_min': (0.1, 1.0),  # meters
            'depth_range_max': (3.0, 10.0),  # meters
            
            # Frame stacking
            'frame_stack_size': (1, self.max_frame_stack),
            'temporal_spacing': (0.05, 0.5),  # seconds
            'stacking_strategy': (0, 3),  # 0-3 maps to enum indices
            'frame_interpolation': (0, 1),  # 0/1 boolean
            
            # Feature extraction
            'spatial_pooling_size': (2, 8),  # kernel size
            'feature_pyramid_levels': (1, 4),
            'attention_mechanism': (0, 1),  # 0/1 boolean
            
            # Proprioceptive fusion weights (will be normalized to sum to 1)
            'imu_weight': (0.0, 1.0),
            'odometry_weight': (0.0, 1.0), 
            'motor_encoder_weight': (0.0, 1.0),
            'fusion_method': (0, 2),  # 0: concat, 1: weighted_sum, 2: attention
            
            # Temporal modeling
            'use_lstm': (0, 1),  # 0/1 boolean
            'lstm_hidden_size': (64, 512),
            'sequence_length': (4, 16),
            
            # Advanced features
            'use_uncertainty_estimation': (0, 1),  # 0/1 boolean
            'confidence_threshold': (0.5, 0.95),
            'failure_detection': (0, 1)  # 0/1 boolean
        }
    
    def _decode_parameters(self, params: Dict[str, float]) -> SensorFusionConfig:
        """Convert optimization parameters to sensor fusion configuration"""
        
        # Preprocessing method
        preprocessing_methods = list(PreprocessingMethod)
        preprocessing_idx = int(round(params['preprocessing_method'])) % len(preprocessing_methods)
        preprocessing_method = preprocessing_methods[preprocessing_idx]
        
        # Normalization method
        normalization_methods = list(NormalizationMethod)
        normalization_idx = int(round(params['normalization_method'])) % len(normalization_methods)
        normalization_method = normalization_methods[normalization_idx]
        
        # Frame stacking strategy
        stacking_strategies = list(FrameStackingStrategy)
        stacking_idx = int(round(params['stacking_strategy'])) % len(stacking_strategies)
        stacking_strategy = stacking_strategies[stacking_idx]
        
        # Fusion method
        fusion_methods = ["concatenate", "weighted_sum", "attention"]
        fusion_idx = int(round(params['fusion_method'])) % len(fusion_methods)
        fusion_method = fusion_methods[fusion_idx]
        
        # Normalize proprioceptive weights to sum to 1
        raw_weights = [
            params['imu_weight'],
            params['odometry_weight'], 
            params['motor_encoder_weight']
        ]
        weight_sum = sum(raw_weights)
        if weight_sum > 0:
            normalized_weights = [w / weight_sum for w in raw_weights]
        else:
            normalized_weights = [1/3, 1/3, 1/3]  # Equal weights fallback
        
        config = SensorFusionConfig(
            # Depth preprocessing
            preprocessing_method=preprocessing_method,
            preprocessing_strength=params['preprocessing_strength'],
            normalization_method=normalization_method,
            depth_range_min=params['depth_range_min'],
            depth_range_max=max(params['depth_range_max'], params['depth_range_min'] + 0.5),  # Ensure max > min
            
            # Frame stacking
            frame_stack_size=max(1, int(round(params['frame_stack_size']))),
            temporal_spacing=params['temporal_spacing'],
            stacking_strategy=stacking_strategy,
            frame_interpolation=params['frame_interpolation'] > 0.5,
            
            # Feature extraction
            spatial_pooling_size=max(2, int(round(params['spatial_pooling_size']))),
            feature_pyramid_levels=max(1, int(round(params['feature_pyramid_levels']))),
            attention_mechanism=params['attention_mechanism'] > 0.5,
            
            # Proprioceptive fusion
            imu_weight=normalized_weights[0],
            odometry_weight=normalized_weights[1],
            motor_encoder_weight=normalized_weights[2],
            fusion_method=fusion_method,
            
            # Temporal modeling
            use_lstm=params['use_lstm'] > 0.5,
            lstm_hidden_size=max(64, int(round(params['lstm_hidden_size']))),
            sequence_length=max(4, int(round(params['sequence_length']))),
            
            # Advanced features
            use_uncertainty_estimation=params['use_uncertainty_estimation'] > 0.5,
            confidence_threshold=params['confidence_threshold'],
            failure_detection=params['failure_detection'] > 0.5
        )
        
        return config
    
    def suggest_configuration(self) -> SensorFusionConfig:
        """Suggest next sensor fusion configuration to evaluate"""
        
        # Use random exploration for initial configurations
        if len(self.performance_history) < 5 or self.gp_model is None:
            return self._random_configuration()
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Use Upper Confidence Bound for exploitation/exploration balance
                acquisition_func = UpperConfidenceBound(self.gp_model, beta=2.0)
                
                # Optimize acquisition function
                candidates, _ = optimize_acqf(
                    acquisition_func,
                    bounds=self.bounds,
                    q=1,
                    num_restarts=20,
                    raw_samples=512,
                )
                
                # Convert to parameter dict
                param_dict = {}
                for i, param_name in enumerate(self.param_names):
                    param_dict[param_name] = candidates[0, i].item()
                
                # Decode to configuration
                config = self._decode_parameters(param_dict)
                
                if self.enable_debug:
                    print(f"[SensorFusionOptimizer] Suggested configuration using BO acquisition")
                    print(f"[SensorFusionOptimizer] Frame stack: {config.frame_stack_size}, "
                          f"Preprocessing: {config.preprocessing_method.value}")
                
                return config
                
        except Exception as e:
            if self.enable_debug:
                print(f"[SensorFusionOptimizer] Acquisition failed: {e}, using random exploration")
            return self._random_configuration()
    
    def _random_configuration(self) -> SensorFusionConfig:
        """Generate random sensor fusion configuration for exploration"""
        
        param_dict = {}
        for param_name in self.param_names:
            min_val, max_val = self.search_space[param_name]
            param_dict[param_name] = np.random.uniform(min_val, max_val)
        
        config = self._decode_parameters(param_dict)
        
        if self.enable_debug:
            print(f"[SensorFusionOptimizer] Generated random configuration")
            print(f"[SensorFusionOptimizer] Frame stack: {config.frame_stack_size}, "
                  f"Preprocessing: {config.preprocessing_method.value}")
        
        return config
    
    def update_performance(self, 
                          config: SensorFusionConfig, 
                          overall_performance: float,
                          performance_components: Optional[Dict[str, float]] = None):
        """Update optimizer with configuration performance"""
        
        # Store configuration and performance
        self.config_history.append(copy.deepcopy(config))
        self.performance_history.append(overall_performance)
        
        # Store performance components if provided
        if performance_components:
            for component_name, value in performance_components.items():
                if component_name in self.performance_components:
                    self.performance_components[component_name].append(value)
        
        # Convert configuration back to parameter tensor for GP
        param_tensor = self._config_to_parameter_tensor(config)
        self.parameter_history.append(param_tensor)
        
        # Update best configuration
        if overall_performance > self.best_performance:
            self.best_performance = overall_performance
            self.best_config = copy.deepcopy(config)
            
            if self.enable_debug:
                print(f"[SensorFusionOptimizer] New best configuration: {overall_performance:.4f}")
                print(f"[SensorFusionOptimizer] Best preprocessing: {config.preprocessing_method.value}")
                print(f"[SensorFusionOptimizer] Best frame stack: {config.frame_stack_size}")
        
        # Update GP model
        self._update_gp_model()
    
    def _config_to_parameter_tensor(self, config: SensorFusionConfig) -> torch.Tensor:
        """Convert sensor fusion configuration back to parameter tensor"""
        
        param_values = []
        for param_name in self.param_names:
            min_val, max_val = self.search_space[param_name]
            
            # Extract corresponding values from configuration
            if param_name == 'preprocessing_method':
                value = list(PreprocessingMethod).index(config.preprocessing_method)
            elif param_name == 'preprocessing_strength':
                value = config.preprocessing_strength
            elif param_name == 'normalization_method':
                value = list(NormalizationMethod).index(config.normalization_method)
            elif param_name == 'depth_range_min':
                value = config.depth_range_min
            elif param_name == 'depth_range_max':
                value = config.depth_range_max
            elif param_name == 'frame_stack_size':
                value = config.frame_stack_size
            elif param_name == 'temporal_spacing':
                value = config.temporal_spacing
            elif param_name == 'stacking_strategy':
                value = list(FrameStackingStrategy).index(config.stacking_strategy)
            elif param_name == 'frame_interpolation':
                value = 1.0 if config.frame_interpolation else 0.0
            elif param_name == 'spatial_pooling_size':
                value = config.spatial_pooling_size
            elif param_name == 'feature_pyramid_levels':
                value = config.feature_pyramid_levels
            elif param_name == 'attention_mechanism':
                value = 1.0 if config.attention_mechanism else 0.0
            elif param_name == 'imu_weight':
                value = config.imu_weight
            elif param_name == 'odometry_weight':
                value = config.odometry_weight
            elif param_name == 'motor_encoder_weight':
                value = config.motor_encoder_weight
            elif param_name == 'fusion_method':
                fusion_methods = ["concatenate", "weighted_sum", "attention"]
                value = fusion_methods.index(config.fusion_method)
            elif param_name == 'use_lstm':
                value = 1.0 if config.use_lstm else 0.0
            elif param_name == 'lstm_hidden_size':
                value = config.lstm_hidden_size
            elif param_name == 'sequence_length':
                value = config.sequence_length
            elif param_name == 'use_uncertainty_estimation':
                value = 1.0 if config.use_uncertainty_estimation else 0.0
            elif param_name == 'confidence_threshold':
                value = config.confidence_threshold
            elif param_name == 'failure_detection':
                value = 1.0 if config.failure_detection else 0.0
            else:
                # Default to middle value
                value = (min_val + max_val) / 2
            
            # Normalize to [0, 1]
            normalized = (value - min_val) / (max_val - min_val)
            normalized = max(0.0, min(1.0, normalized))  # Clamp to bounds
            param_values.append(normalized)
        
        return torch.tensor(param_values, dtype=torch.float64)
    
    def _update_gp_model(self):
        """Update GP model with sensor fusion performance data"""
        
        if len(self.performance_history) < 3:
            return
        
        try:
            X = torch.stack(self.parameter_history)
            y = torch.tensor(self.performance_history, dtype=torch.float64).unsqueeze(-1)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                self.gp_model = SingleTaskGP(X, y)
                self.mll = ExactMarginalLogLikelihood(self.gp_model.likelihood, self.gp_model)
                fit_gpytorch_mll(self.mll)
                
                if self.enable_debug:
                    print(f"[SensorFusionOptimizer] Updated GP model with {len(self.performance_history)} configurations")
        
        except Exception as e:
            if self.enable_debug:
                print(f"[SensorFusionOptimizer] Failed to update GP model: {e}")
            self.gp_model = None
    
    def get_best_configuration(self) -> Optional[SensorFusionConfig]:
        """Get the best sensor fusion configuration found so far"""
        return self.best_config
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get sensor fusion optimization statistics"""
        
        stats = {
            'n_evaluated_configurations': len(self.config_history),
            'best_performance': self.best_performance,
            'performance_improvement': 0.0,
            'preprocessing_method_distribution': {},
            'frame_stack_size_distribution': {},
            'temporal_modeling_usage': 0.0
        }
        
        if len(self.performance_history) >= 5:
            # Calculate improvement
            early_performance = np.mean(self.performance_history[:3])
            recent_performance = np.mean(self.performance_history[-3:])
            stats['performance_improvement'] = recent_performance - early_performance
        
        # Analyze configuration distribution
        if self.config_history:
            # Preprocessing methods tried
            preprocessing_counts = {}
            for config in self.config_history:
                method = config.preprocessing_method.value
                preprocessing_counts[method] = preprocessing_counts.get(method, 0) + 1
            stats['preprocessing_method_distribution'] = preprocessing_counts
            
            # Frame stack sizes tried
            frame_stack_counts = {}
            for config in self.config_history:
                size = config.frame_stack_size
                frame_stack_counts[size] = frame_stack_counts.get(size, 0) + 1
            stats['frame_stack_size_distribution'] = frame_stack_counts
            
            # LSTM usage rate
            lstm_usage = sum(1 for config in self.config_history if config.use_lstm)
            stats['temporal_modeling_usage'] = lstm_usage / len(self.config_history)
        
        # Component performance trends
        for component_name, values in self.performance_components.items():
            if values:
                stats[f'{component_name}_mean'] = np.mean(values)
                stats[f'{component_name}_trend'] = np.mean(values[-3:]) - np.mean(values[:3]) if len(values) >= 6 else 0.0
        
        return stats
    
    def apply_configuration(self, config: SensorFusionConfig) -> Dict[str, Any]:
        """
        Apply sensor fusion configuration and return processing parameters.
        
        This would integrate with the actual sensor processing pipeline.
        """
        
        processing_params = {
            # Depth preprocessing parameters
            'depth_preprocessing': {
                'method': config.preprocessing_method.value,
                'strength': config.preprocessing_strength,
                'normalization': config.normalization_method.value,
                'range_min': config.depth_range_min,
                'range_max': config.depth_range_max
            },
            
            # Frame stacking parameters
            'frame_stacking': {
                'stack_size': config.frame_stack_size,
                'temporal_spacing': config.temporal_spacing,
                'strategy': config.stacking_strategy.value,
                'interpolation': config.frame_interpolation
            },
            
            # Feature extraction parameters
            'feature_extraction': {
                'pooling_size': config.spatial_pooling_size,
                'pyramid_levels': config.feature_pyramid_levels,
                'attention': config.attention_mechanism
            },
            
            # Sensor fusion parameters
            'sensor_fusion': {
                'imu_weight': config.imu_weight,
                'odometry_weight': config.odometry_weight,
                'encoder_weight': config.motor_encoder_weight,
                'fusion_method': config.fusion_method
            },
            
            # Temporal modeling parameters
            'temporal_modeling': {
                'use_lstm': config.use_lstm,
                'hidden_size': config.lstm_hidden_size,
                'sequence_length': config.sequence_length
            },
            
            # Advanced parameters
            'advanced': {
                'uncertainty_estimation': config.use_uncertainty_estimation,
                'confidence_threshold': config.confidence_threshold,
                'failure_detection': config.failure_detection
            }
        }
        
        if self.enable_debug:
            print(f"[SensorFusionOptimizer] Applied configuration:")
            print(f"  Preprocessing: {config.preprocessing_method.value} (strength: {config.preprocessing_strength:.2f})")
            print(f"  Frame stack: {config.frame_stack_size} frames, {config.temporal_spacing:.3f}s spacing")
            print(f"  Fusion weights: IMU={config.imu_weight:.2f}, Odom={config.odometry_weight:.2f}, Enc={config.motor_encoder_weight:.2f}")
            print(f"  Temporal modeling: {'LSTM' if config.use_lstm else 'None'}")
        
        return processing_params
    
    def save_optimizer_state(self, filepath: str):
        """Save optimizer state to file"""
        try:
            state = {
                'search_space': self.search_space,
                'config_history': [
                    {
                        'preprocessing_method': config.preprocessing_method.value,
                        'preprocessing_strength': config.preprocessing_strength,
                        'normalization_method': config.normalization_method.value,
                        'depth_range_min': config.depth_range_min,
                        'depth_range_max': config.depth_range_max,
                        'frame_stack_size': config.frame_stack_size,
                        'temporal_spacing': config.temporal_spacing,
                        'stacking_strategy': config.stacking_strategy.value,
                        'frame_interpolation': config.frame_interpolation,
                        'spatial_pooling_size': config.spatial_pooling_size,
                        'feature_pyramid_levels': config.feature_pyramid_levels,
                        'attention_mechanism': config.attention_mechanism,
                        'imu_weight': config.imu_weight,
                        'odometry_weight': config.odometry_weight,
                        'motor_encoder_weight': config.motor_encoder_weight,
                        'fusion_method': config.fusion_method,
                        'use_lstm': config.use_lstm,
                        'lstm_hidden_size': config.lstm_hidden_size,
                        'sequence_length': config.sequence_length,
                        'use_uncertainty_estimation': config.use_uncertainty_estimation,
                        'confidence_threshold': config.confidence_threshold,
                        'failure_detection': config.failure_detection
                    } for config in self.config_history
                ],
                'performance_history': self.performance_history,
                'performance_components': self.performance_components,
                'best_performance': self.best_performance
            }
            
            torch.save(state, filepath)
            
            if self.enable_debug:
                print(f"[SensorFusionOptimizer] Saved optimizer state to {filepath}")
        
        except Exception as e:
            if self.enable_debug:
                print(f"[SensorFusionOptimizer] Failed to save state: {e}")