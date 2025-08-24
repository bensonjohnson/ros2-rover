#!/usr/bin/env python3
"""
Bayesian Neural Architecture Search (NAS) for Rover Depth Image Processing
Automatically optimizes neural network architectures using Bayesian optimization
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union
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

class ActivationType(Enum):
    """Supported activation function types"""
    RELU = "relu"
    LEAKY_RELU = "leaky_relu" 
    ELU = "elu"
    GELU = "gelu"
    SWISH = "swish"
    MISH = "mish"

class ConvBlockType(Enum):
    """Types of convolutional blocks"""
    BASIC = "basic"           # Simple Conv2d + BatchNorm + Activation
    RESIDUAL = "residual"     # ResNet-style residual block
    DENSE = "dense"           # DenseNet-style dense block
    SEPARABLE = "separable"   # Depthwise separable convolution
    INVERTED = "inverted"     # MobileNet-style inverted residual

@dataclass 
class ConvLayerConfig:
    """Configuration for a convolutional layer"""
    out_channels: int
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1
    block_type: ConvBlockType = ConvBlockType.BASIC
    activation: ActivationType = ActivationType.RELU
    dropout_rate: float = 0.0
    use_batch_norm: bool = True

@dataclass
class FusionLayerConfig:
    """Configuration for sensor fusion layers"""
    hidden_sizes: List[int]
    activation: ActivationType = ActivationType.RELU
    dropout_rate: float = 0.3
    fusion_method: str = "concatenate"  # concatenate, add, multiply, attention

@dataclass
class NetworkArchitecture:
    """Complete neural network architecture specification"""
    # Depth processing branch
    depth_conv_layers: List[ConvLayerConfig]
    depth_fc_size: int
    
    # Sensor processing branch  
    sensor_fc_layers: List[int]
    sensor_activation: ActivationType
    
    # Fusion configuration
    fusion_config: FusionLayerConfig
    
    # Output configuration
    output_size: int = 3
    
    # Metadata
    estimated_parameters: int = 0
    estimated_flops: int = 0
    architecture_id: str = ""

class BayesianArchitectureOptimizer:
    """
    Bayesian optimization for neural network architecture search.
    
    Optimizes network topology for depth image processing tasks by searching over:
    - Convolutional layer configurations
    - Layer depths and channel progressions
    - Activation functions and regularization
    - Sensor fusion architectures
    """
    
    def __init__(self,
                 input_shape: Tuple[int, int, int] = (1, 160, 288),  # (C, H, W) for depth images
                 sensor_input_size: int = 16,
                 output_size: int = 3,
                 max_parameters: int = 5000000,  # 5M parameter budget
                 enable_debug: bool = False):
        
        if not BOTORCH_AVAILABLE:
            raise ImportError("BoTorch is required for Neural Architecture Search. Install with: pip install botorch")
        
        self.input_shape = input_shape
        self.sensor_input_size = sensor_input_size
        self.output_size = output_size
        self.max_parameters = max_parameters
        self.enable_debug = enable_debug
        
        # Architecture search space definition
        self.search_space = self._define_search_space()
        self.param_names = list(self.search_space.keys())
        self.n_params = len(self.param_names)
        
        # Convert bounds to torch tensors
        bounds_array = np.array([self.search_space[name] for name in self.param_names])
        self.bounds = torch.tensor(bounds_array.T, dtype=torch.float64)
        
        # Architecture evaluation history
        self.architecture_history: List[NetworkArchitecture] = []
        self.performance_history: List[float] = []
        self.parameter_history: List[torch.Tensor] = []
        
        # Bayesian optimization components
        self.gp_model: Optional[SingleTaskGP] = None
        self.mll = None
        
        # Architecture generation
        self.architecture_counter = 0
        
        # Performance tracking
        self.best_architecture: Optional[NetworkArchitecture] = None
        self.best_performance: float = -float('inf')
        
        if self.enable_debug:
            print(f"[BayesianNAS] Initialized with input shape {input_shape}")
            print(f"[BayesianNAS] Search space: {len(self.search_space)} parameters")
            print(f"[BayesianNAS] Parameter budget: {max_parameters:,}")
    
    def _define_search_space(self) -> Dict[str, Tuple[float, float]]:
        """Define the neural architecture search space"""
        
        search_space = {
            # Depth convolutional branch architecture
            'n_conv_layers': (3, 8),                    # Number of conv layers
            'base_channels': (16, 128),                 # Starting channel count
            'channel_multiplier': (1.5, 3.0),          # Channel growth factor
            'conv_kernel_sizes': (3, 7),               # Kernel sizes (will be discretized)
            'conv_stride_pattern': (1, 3),             # Stride pattern (1=conservative, 3=aggressive)
            
            # Activation functions (encoded as integers)
            'depth_activation': (0, 5),                # Index into ActivationType enum
            'sensor_activation': (0, 5),               # Index into ActivationType enum
            
            # Regularization
            'conv_dropout_rate': (0.0, 0.5),          # Dropout in conv layers
            'fc_dropout_rate': (0.1, 0.7),            # Dropout in FC layers
            'use_batch_norm': (0, 1),                 # Whether to use batch norm
            
            # Sensor fusion branch
            'sensor_n_layers': (2, 5),                # Number of sensor FC layers
            'sensor_layer_size_multiplier': (0.5, 4.0), # Size multiplier for sensor layers
            
            # Fusion configuration
            'fusion_hidden_size': (128, 1024),        # Fusion layer size
            'fusion_n_layers': (1, 4),                # Number of fusion layers
            'fusion_method': (0, 3),                  # Fusion method (concatenate, add, etc.)
            
            # Architecture variants
            'use_residual_blocks': (0, 1),            # Whether to use residual connections
            'use_separable_conv': (0, 1),             # Whether to use separable convolutions
            'depth_branch_width_factor': (0.5, 2.0),  # Width scaling for depth branch
        }
        
        return search_space
    
    def _decode_architecture_parameters(self, params: Dict[str, float]) -> NetworkArchitecture:
        """Decode optimization parameters into a concrete network architecture"""
        
        # Depth convolutional layers
        n_conv_layers = int(round(params['n_conv_layers']))
        base_channels = int(round(params['base_channels']))
        channel_multiplier = params['channel_multiplier']
        kernel_size = int(round(params['conv_kernel_sizes']))
        if kernel_size % 2 == 0:  # Ensure odd kernel size
            kernel_size += 1
        
        use_batch_norm = params['use_batch_norm'] > 0.5
        use_residual = params['use_residual_blocks'] > 0.5
        use_separable = params['use_separable_conv'] > 0.5
        conv_dropout = params['conv_dropout_rate']
        
        # Determine stride pattern
        stride_pattern_idx = int(round(params['conv_stride_pattern']))
        stride_patterns = {
            1: [1, 2, 1, 2, 1, 2, 1, 2],  # Conservative: stride 2 every other layer
            2: [2, 2, 2, 1, 1, 1, 1, 1],  # Moderate: early downsampling
            3: [2, 2, 2, 2, 1, 1, 1, 1],  # Aggressive: heavy early downsampling
        }
        strides = stride_patterns.get(stride_pattern_idx, stride_patterns[2])
        
        # Build convolutional layers
        depth_conv_layers = []
        current_channels = self.input_shape[0]  # Start with input channels
        
        for i in range(n_conv_layers):
            out_channels = int(base_channels * (channel_multiplier ** i))
            # Apply width factor
            out_channels = int(out_channels * params['depth_branch_width_factor'])
            out_channels = max(8, min(512, out_channels))  # Clamp to reasonable range
            
            stride = strides[i] if i < len(strides) else 1
            
            # Determine block type
            if use_residual and i > 0:  # Skip residual for first layer
                block_type = ConvBlockType.RESIDUAL
            elif use_separable:
                block_type = ConvBlockType.SEPARABLE
            else:
                block_type = ConvBlockType.BASIC
            
            # Activation type
            activation_idx = int(round(params['depth_activation']))
            activation = list(ActivationType)[activation_idx % len(ActivationType)]
            
            layer_config = ConvLayerConfig(
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                block_type=block_type,
                activation=activation,
                dropout_rate=conv_dropout if i > n_conv_layers // 2 else 0.0,  # Dropout in later layers
                use_batch_norm=use_batch_norm
            )
            
            depth_conv_layers.append(layer_config)
            current_channels = out_channels
        
        # Calculate depth FC size based on final conv output
        depth_fc_size = self._estimate_conv_output_size(depth_conv_layers)
        
        # Sensor processing branch
        n_sensor_layers = int(round(params['sensor_n_layers']))
        sensor_layer_multiplier = params['sensor_layer_size_multiplier']
        sensor_activation_idx = int(round(params['sensor_activation']))
        sensor_activation = list(ActivationType)[sensor_activation_idx % len(ActivationType)]
        
        # Build sensor FC layers with decreasing sizes
        sensor_fc_layers = []
        current_size = self.sensor_input_size
        for i in range(n_sensor_layers):
            # Exponential decay in layer size
            next_size = int(current_size * sensor_layer_multiplier * (0.7 ** i))
            next_size = max(32, min(512, next_size))  # Clamp to reasonable range
            sensor_fc_layers.append(next_size)
            current_size = next_size
        
        # Fusion configuration
        fusion_hidden_size = int(round(params['fusion_hidden_size']))
        fusion_n_layers = int(round(params['fusion_n_layers']))
        fusion_method_idx = int(round(params['fusion_method']))
        fusion_methods = ['concatenate', 'add', 'multiply', 'attention']
        fusion_method = fusion_methods[fusion_method_idx % len(fusion_methods)]
        
        fusion_hidden_sizes = []
        current_fusion_size = fusion_hidden_size
        for i in range(fusion_n_layers):
            fusion_hidden_sizes.append(current_fusion_size)
            current_fusion_size = int(current_fusion_size * 0.8)  # Decrease size
        
        fusion_config = FusionLayerConfig(
            hidden_sizes=fusion_hidden_sizes,
            activation=sensor_activation,  # Use same as sensor branch
            dropout_rate=params['fc_dropout_rate'],
            fusion_method=fusion_method
        )
        
        # Create complete architecture
        architecture = NetworkArchitecture(
            depth_conv_layers=depth_conv_layers,
            depth_fc_size=depth_fc_size,
            sensor_fc_layers=sensor_fc_layers,
            sensor_activation=sensor_activation,
            fusion_config=fusion_config,
            output_size=self.output_size,
            architecture_id=f"nas_arch_{self.architecture_counter}"
        )
        
        # Estimate parameter count and FLOPs
        architecture.estimated_parameters = self._estimate_parameter_count(architecture)
        architecture.estimated_flops = self._estimate_flops(architecture)
        
        self.architecture_counter += 1
        
        return architecture
    
    def _estimate_conv_output_size(self, conv_layers: List[ConvLayerConfig]) -> int:
        """Estimate the output size after all conv layers for FC connection"""
        
        h, w = self.input_shape[1], self.input_shape[2]  # Start with input spatial dims
        channels = conv_layers[-1].out_channels if conv_layers else self.input_shape[0]
        
        # Simulate conv layer size reductions
        for layer in conv_layers:
            # Simple size calculation (assumes proper padding)
            h = h // layer.stride
            w = w // layer.stride
            # Add pooling effect if stride > 1
            if layer.stride > 1:
                h = max(1, h)
                w = max(1, w)
        
        # Add global average pooling or flattening size
        # For now, assume we flatten the final conv output
        total_size = channels * h * w
        
        # Cap the FC size to reasonable limits
        max_fc_size = 2048
        return min(total_size, max_fc_size)
    
    def _estimate_parameter_count(self, architecture: NetworkArchitecture) -> int:
        """Estimate total parameter count for architecture"""
        
        total_params = 0
        
        # Depth conv layers
        prev_channels = self.input_shape[0]
        for layer in architecture.depth_conv_layers:
            # Conv layer parameters
            conv_params = prev_channels * layer.out_channels * (layer.kernel_size ** 2)
            
            # Batch norm parameters (if used)
            bn_params = layer.out_channels * 2 if layer.use_batch_norm else 0
            
            total_params += conv_params + bn_params
            prev_channels = layer.out_channels
        
        # Depth FC layer
        total_params += architecture.depth_fc_size * 512  # Assume 512 is target depth FC output
        
        # Sensor FC layers
        prev_size = self.sensor_input_size
        for layer_size in architecture.sensor_fc_layers:
            total_params += prev_size * layer_size + layer_size  # weights + biases
            prev_size = layer_size
        
        # Fusion layers
        fusion_input_size = 512 + prev_size  # Depth FC + Sensor FC output
        for hidden_size in architecture.fusion_config.hidden_sizes:
            total_params += fusion_input_size * hidden_size + hidden_size
            fusion_input_size = hidden_size
        
        # Output layer
        total_params += fusion_input_size * architecture.output_size + architecture.output_size
        
        return total_params
    
    def _estimate_flops(self, architecture: NetworkArchitecture) -> int:
        """Estimate FLOPs (floating point operations) for architecture"""
        
        total_flops = 0
        h, w = self.input_shape[1], self.input_shape[2]
        
        # Conv layer FLOPs
        prev_channels = self.input_shape[0]
        for layer in architecture.depth_conv_layers:
            # Conv FLOPs: output_size * kernel_ops
            conv_flops = (h * w) * layer.out_channels * prev_channels * (layer.kernel_size ** 2)
            
            # Account for stride reduction in output size
            h = h // layer.stride
            w = w // layer.stride
            
            total_flops += conv_flops
            prev_channels = layer.out_channels
        
        # FC layer FLOPs (matrix multiplications)
        total_flops += architecture.depth_fc_size * 512
        
        # Sensor FC FLOPs
        prev_size = self.sensor_input_size
        for layer_size in architecture.sensor_fc_layers:
            total_flops += prev_size * layer_size
            prev_size = layer_size
        
        # Fusion FLOPs
        fusion_input_size = 512 + prev_size
        for hidden_size in architecture.fusion_config.hidden_sizes:
            total_flops += fusion_input_size * hidden_size
            fusion_input_size = hidden_size
        
        # Output layer FLOPs
        total_flops += fusion_input_size * architecture.output_size
        
        return total_flops
    
    def suggest_architecture(self) -> NetworkArchitecture:
        """Suggest next architecture to evaluate using Bayesian optimization"""
        
        # Use random exploration for initial architectures
        if len(self.performance_history) < 5 or self.gp_model is None:
            return self._random_architecture()
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Use Upper Confidence Bound for architecture search
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
                
                # Decode to architecture
                architecture = self._decode_architecture_parameters(param_dict)
                
                # Check parameter budget constraint
                if architecture.estimated_parameters > self.max_parameters:
                    # Try to scale down the architecture
                    architecture = self._scale_down_architecture(architecture)
                
                if self.enable_debug:
                    print(f"[BayesianNAS] Suggested architecture with {architecture.estimated_parameters:,} parameters")
                
                return architecture
                
        except Exception as e:
            if self.enable_debug:
                print(f"[BayesianNAS] Acquisition failed: {e}, using random exploration")
            return self._random_architecture()
    
    def _random_architecture(self) -> NetworkArchitecture:
        """Generate random architecture for exploration"""
        
        param_dict = {}
        for param_name in self.param_names:
            min_val, max_val = self.search_space[param_name]
            param_dict[param_name] = np.random.uniform(min_val, max_val)
        
        architecture = self._decode_architecture_parameters(param_dict)
        
        # Ensure parameter budget is respected
        if architecture.estimated_parameters > self.max_parameters:
            architecture = self._scale_down_architecture(architecture)
        
        if self.enable_debug:
            print(f"[BayesianNAS] Random architecture with {architecture.estimated_parameters:,} parameters")
        
        return architecture
    
    def _scale_down_architecture(self, architecture: NetworkArchitecture) -> NetworkArchitecture:
        """Scale down architecture to meet parameter budget"""
        
        # Simple scaling: reduce channel counts and layer sizes
        scaling_factor = min(0.8, (self.max_parameters / architecture.estimated_parameters) ** 0.5)
        
        # Scale conv layers
        for layer in architecture.depth_conv_layers:
            layer.out_channels = max(8, int(layer.out_channels * scaling_factor))
        
        # Scale FC layers
        architecture.sensor_fc_layers = [max(32, int(size * scaling_factor)) 
                                       for size in architecture.sensor_fc_layers]
        
        # Scale fusion layers
        architecture.fusion_config.hidden_sizes = [max(64, int(size * scaling_factor)) 
                                                   for size in architecture.fusion_config.hidden_sizes]
        
        # Recalculate parameter count
        architecture.estimated_parameters = self._estimate_parameter_count(architecture)
        architecture.estimated_flops = self._estimate_flops(architecture)
        
        return architecture
    
    def update_performance(self, architecture: NetworkArchitecture, performance: float):
        """Update Bayesian optimizer with architecture performance"""
        
        # Store architecture and performance
        self.architecture_history.append(architecture)
        self.performance_history.append(performance)
        
        # Convert architecture back to parameter tensor for GP
        param_tensor = self._architecture_to_parameter_tensor(architecture)
        self.parameter_history.append(param_tensor)
        
        # Update best architecture
        if performance > self.best_performance:
            self.best_performance = performance
            self.best_architecture = copy.deepcopy(architecture)
            
            if self.enable_debug:
                print(f"[BayesianNAS] New best architecture: {performance:.4f}")
                print(f"[BayesianNAS] Parameters: {architecture.estimated_parameters:,}")
        
        # Update GP model
        self._update_gp_model()
    
    def _architecture_to_parameter_tensor(self, architecture: NetworkArchitecture) -> torch.Tensor:
        """Convert architecture back to normalized parameter tensor"""
        
        # This is a simplified reverse mapping
        # In practice, would need to store the original parameters used to generate each architecture
        
        param_values = []
        for param_name in self.param_names:
            min_val, max_val = self.search_space[param_name]
            
            # Extract corresponding values from architecture (simplified)
            if param_name == 'n_conv_layers':
                value = len(architecture.depth_conv_layers)
            elif param_name == 'base_channels':
                value = architecture.depth_conv_layers[0].out_channels if architecture.depth_conv_layers else 32
            elif param_name == 'fusion_hidden_size':
                value = architecture.fusion_config.hidden_sizes[0] if architecture.fusion_config.hidden_sizes else 256
            else:
                # Default to middle value for other parameters
                value = (min_val + max_val) / 2
            
            # Normalize to [0, 1]
            normalized = (value - min_val) / (max_val - min_val)
            normalized = max(0.0, min(1.0, normalized))  # Clamp to bounds
            param_values.append(normalized)
        
        return torch.tensor(param_values, dtype=torch.float64)
    
    def _update_gp_model(self):
        """Update GP model with architecture performance data"""
        
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
                    print(f"[BayesianNAS] Updated GP model with {len(self.performance_history)} architectures")
        
        except Exception as e:
            if self.enable_debug:
                print(f"[BayesianNAS] Failed to update GP model: {e}")
            self.gp_model = None
    
    def get_best_architecture(self) -> Optional[NetworkArchitecture]:
        """Get the best architecture found so far"""
        return self.best_architecture
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get architecture search statistics"""
        
        stats = {
            'n_evaluated_architectures': len(self.architecture_history),
            'best_performance': self.best_performance,
            'performance_improvement': 0.0,
            'parameter_efficiency': 0.0,
            'architecture_diversity': 0.0
        }
        
        if len(self.performance_history) >= 5:
            # Calculate improvement
            early_performance = np.mean(self.performance_history[:3])
            recent_performance = np.mean(self.performance_history[-3:])
            stats['performance_improvement'] = recent_performance - early_performance
            
            # Parameter efficiency: performance per million parameters
            if self.best_architecture:
                stats['parameter_efficiency'] = self.best_performance / (self.best_architecture.estimated_parameters / 1e6)
            
            # Architecture diversity: coefficient of variation in parameter counts
            param_counts = [arch.estimated_parameters for arch in self.architecture_history]
            if param_counts:
                mean_params = np.mean(param_counts)
                std_params = np.std(param_counts)
                stats['architecture_diversity'] = std_params / mean_params if mean_params > 0 else 0
        
        return stats