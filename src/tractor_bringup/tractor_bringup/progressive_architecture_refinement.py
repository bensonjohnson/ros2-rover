#!/usr/bin/env python3
"""
Progressive Architecture Refinement for Neural Architecture Search
Gradually evolves network architectures from simple to complex during training
"""

import numpy as np
import copy
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import time

from .bayesian_nas import NetworkArchitecture, ConvLayerConfig, FusionLayerConfig, ActivationType, ConvBlockType

class ProgressionStage(Enum):
    """Stages of progressive architecture refinement"""
    INITIALIZATION = "initialization"     # Start with minimal architecture
    EXPANSION = "expansion"               # Gradually add complexity
    REFINEMENT = "refinement"             # Fine-tune existing structure  
    SPECIALIZATION = "specialization"     # Optimize for specific tasks
    MATURATION = "maturation"             # Final optimizations

@dataclass
class ProgressionConfig:
    """Configuration for progressive refinement"""
    # Stage transition criteria
    performance_threshold_increase: float = 0.05    # Performance improvement needed to advance
    min_stage_duration: int = 500                   # Minimum training steps per stage
    max_stage_duration: int = 2000                  # Maximum training steps per stage
    convergence_patience: int = 200                 # Steps without improvement before transition
    
    # Complexity constraints per stage
    stage_parameter_budgets: Dict[ProgressionStage, int] = None
    stage_complexity_factors: Dict[ProgressionStage, float] = None
    
    def __post_init__(self):
        if self.stage_parameter_budgets is None:
            self.stage_parameter_budgets = {
                ProgressionStage.INITIALIZATION: 50000,     # 50K parameters
                ProgressionStage.EXPANSION: 200000,         # 200K parameters  
                ProgressionStage.REFINEMENT: 500000,        # 500K parameters
                ProgressionStage.SPECIALIZATION: 1000000,   # 1M parameters
                ProgressionStage.MATURATION: 2000000        # 2M parameters
            }
        
        if self.stage_complexity_factors is None:
            self.stage_complexity_factors = {
                ProgressionStage.INITIALIZATION: 0.3,       # Very simple
                ProgressionStage.EXPANSION: 0.5,            # Simple to moderate
                ProgressionStage.REFINEMENT: 0.7,           # Moderate complexity
                ProgressionStage.SPECIALIZATION: 0.9,       # High complexity
                ProgressionStage.MATURATION: 1.0            # Full complexity
            }

@dataclass
class ProgressionState:
    """Current state of progressive refinement"""
    current_stage: ProgressionStage
    stage_start_step: int
    steps_in_stage: int
    best_performance_in_stage: float
    steps_since_improvement: int
    stage_architectures: List[NetworkArchitecture]
    performance_history: List[float]

class ProgressiveArchitectureRefinement:
    """
    Manages progressive refinement of neural network architectures.
    
    Starts with simple architectures and gradually increases complexity based on
    performance improvements and training stability.
    """
    
    def __init__(self, 
                 initial_architecture: NetworkArchitecture,
                 progression_config: Optional[ProgressionConfig] = None,
                 enable_debug: bool = False):
        
        self.initial_architecture = copy.deepcopy(initial_architecture)
        self.config = progression_config or ProgressionConfig()
        self.enable_debug = enable_debug
        
        # Initialize progression state
        self.state = ProgressionState(
            current_stage=ProgressionStage.INITIALIZATION,
            stage_start_step=0,
            steps_in_stage=0,
            best_performance_in_stage=-float('inf'),
            steps_since_improvement=0,
            stage_architectures=[],
            performance_history=[]
        )
        
        # Architecture evolution tracking
        self.architecture_lineage: List[Tuple[NetworkArchitecture, ProgressionStage, float]] = []
        self.current_architecture = copy.deepcopy(initial_architecture)
        
        # Refinement strategies for each stage
        self.refinement_strategies = {
            ProgressionStage.INITIALIZATION: self._initialization_strategy,
            ProgressionStage.EXPANSION: self._expansion_strategy,
            ProgressionStage.REFINEMENT: self._refinement_strategy,
            ProgressionStage.SPECIALIZATION: self._specialization_strategy,
            ProgressionStage.MATURATION: self._maturation_strategy
        }
        
        if self.enable_debug:
            print(f"[ProgressiveNAS] Initialized in {self.state.current_stage.value} stage")
            print(f"[ProgressiveNAS] Parameter budget: {self.config.stage_parameter_budgets[self.state.current_stage]:,}")
    
    def update_progress(self, training_step: int, performance: float, 
                       current_architecture: Optional[NetworkArchitecture] = None) -> bool:
        """
        Update progressive refinement state and check for stage transitions.
        
        Returns:
            True if stage transition occurred, False otherwise
        """
        
        if current_architecture:
            self.current_architecture = current_architecture
        
        # Update state
        self.state.steps_in_stage = training_step - self.state.stage_start_step
        self.state.performance_history.append(performance)
        
        # Track performance improvements
        if performance > self.state.best_performance_in_stage:
            self.state.best_performance_in_stage = performance
            self.state.steps_since_improvement = 0
            
            # Record this architecture as successful for current stage
            if current_architecture:
                architecture_copy = copy.deepcopy(current_architecture)
                self.state.stage_architectures.append(architecture_copy)
                self.architecture_lineage.append((architecture_copy, self.state.current_stage, performance))
        else:
            self.state.steps_since_improvement += 1
        
        # Check for stage transition
        if self._should_transition_stage():
            self._transition_to_next_stage(training_step)
            return True
        
        return False
    
    def _should_transition_stage(self) -> bool:
        """Determine if we should transition to the next stage"""
        
        # Don't transition if we haven't met minimum duration
        if self.state.steps_in_stage < self.config.min_stage_duration:
            return False
        
        # Force transition if maximum duration exceeded
        if self.state.steps_in_stage >= self.config.max_stage_duration:
            return True
        
        # Transition if performance has converged (no improvement for a while)
        if self.state.steps_since_improvement >= self.config.convergence_patience:
            return True
        
        # Transition if we've achieved sufficient performance improvement
        if len(self.state.performance_history) >= 20:
            recent_avg = np.mean(self.state.performance_history[-10:])
            early_avg = np.mean(self.state.performance_history[:10])
            improvement = recent_avg - early_avg
            
            if improvement >= self.config.performance_threshold_increase:
                return True
        
        return False
    
    def _transition_to_next_stage(self, training_step: int):
        """Transition to the next progression stage"""
        
        # Determine next stage
        current_stages = list(ProgressionStage)
        current_idx = current_stages.index(self.state.current_stage)
        
        if current_idx < len(current_stages) - 1:
            next_stage = current_stages[current_idx + 1]
        else:
            # Already at final stage
            return
        
        if self.enable_debug:
            print(f"[ProgressiveNAS] Transitioning from {self.state.current_stage.value} to {next_stage.value}")
            print(f"[ProgressiveNAS] Best performance in stage: {self.state.best_performance_in_stage:.4f}")
            print(f"[ProgressiveNAS] Steps in stage: {self.state.steps_in_stage}")
        
        # Update state for new stage
        self.state.current_stage = next_stage
        self.state.stage_start_step = training_step
        self.state.steps_in_stage = 0
        self.state.best_performance_in_stage = -float('inf')
        self.state.steps_since_improvement = 0
        self.state.stage_architectures = []
        self.state.performance_history = []
        
        # Generate new architecture for this stage
        self.current_architecture = self._evolve_architecture_for_stage(next_stage)
        
        if self.enable_debug:
            print(f"[ProgressiveNAS] New parameter budget: {self.config.stage_parameter_budgets[next_stage]:,}")
            print(f"[ProgressiveNAS] Evolved architecture with {self.current_architecture.estimated_parameters:,} parameters")
    
    def _evolve_architecture_for_stage(self, stage: ProgressionStage) -> NetworkArchitecture:
        """Evolve current architecture for the new stage"""
        
        # Get best architecture from previous stage as starting point
        base_architecture = self.current_architecture
        if self.state.stage_architectures:
            # Use best performing architecture from previous stage
            best_prev_arch = max(self.architecture_lineage, key=lambda x: x[2])[0]
            base_architecture = best_prev_arch
        
        # Apply stage-specific refinement strategy
        strategy_func = self.refinement_strategies.get(stage, self._default_strategy)
        evolved_architecture = strategy_func(base_architecture)
        
        return evolved_architecture
    
    def _initialization_strategy(self, base_architecture: NetworkArchitecture) -> NetworkArchitecture:
        """Strategy for initialization stage: Keep architecture minimal and stable"""
        
        arch = copy.deepcopy(base_architecture)
        
        # Ensure minimal architecture
        target_params = self.config.stage_parameter_budgets[ProgressionStage.INITIALIZATION]
        
        # Reduce conv layers if needed
        while len(arch.depth_conv_layers) > 3 and self._estimate_parameters(arch) > target_params:
            arch.depth_conv_layers.pop()
        
        # Reduce channel counts
        for layer in arch.depth_conv_layers:
            layer.out_channels = min(layer.out_channels, 64)
            layer.block_type = ConvBlockType.BASIC  # Use simple blocks
            layer.dropout_rate = max(0.0, layer.dropout_rate - 0.1)  # Less regularization
        
        # Simplify fusion
        arch.fusion_config.hidden_sizes = arch.fusion_config.hidden_sizes[:2]  # Max 2 layers
        arch.fusion_config.hidden_sizes = [min(size, 256) for size in arch.fusion_config.hidden_sizes]
        
        arch.estimated_parameters = self._estimate_parameters(arch)
        return arch
    
    def _expansion_strategy(self, base_architecture: NetworkArchitecture) -> NetworkArchitecture:
        """Strategy for expansion stage: Gradually add complexity"""
        
        arch = copy.deepcopy(base_architecture)
        target_params = self.config.stage_parameter_budgets[ProgressionStage.EXPANSION]
        
        # Add conv layers if budget allows
        if len(arch.depth_conv_layers) < 5:
            last_layer = arch.depth_conv_layers[-1] if arch.depth_conv_layers else None
            if last_layer:
                new_layer = copy.deepcopy(last_layer)
                new_layer.out_channels = min(last_layer.out_channels * 2, 128)
                new_layer.stride = 1  # Maintain spatial resolution
                arch.depth_conv_layers.append(new_layer)
        
        # Increase channel counts gradually
        for i, layer in enumerate(arch.depth_conv_layers):
            max_channels = min(128, 32 * (2 ** i))
            layer.out_channels = min(max_channels, int(layer.out_channels * 1.2))
        
        # Expand fusion layers
        if len(arch.fusion_config.hidden_sizes) < 3:
            arch.fusion_config.hidden_sizes.append(256)
        
        # Scale down if over budget
        arch = self._scale_to_budget(arch, target_params)
        return arch
    
    def _refinement_strategy(self, base_architecture: NetworkArchitecture) -> NetworkArchitecture:
        """Strategy for refinement stage: Optimize existing structure"""
        
        arch = copy.deepcopy(base_architecture)
        target_params = self.config.stage_parameter_budgets[ProgressionStage.REFINEMENT]
        
        # Introduce more sophisticated blocks
        for i, layer in enumerate(arch.depth_conv_layers):
            if i > 0 and layer.block_type == ConvBlockType.BASIC:
                layer.block_type = ConvBlockType.RESIDUAL  # Add residual connections
            
            # Optimize kernel sizes
            if layer.kernel_size == 3:
                layer.kernel_size = 5 if i < len(arch.depth_conv_layers) // 2 else 3
                layer.padding = layer.kernel_size // 2
        
        # Add more fusion complexity
        arch.fusion_config.fusion_method = "attention"  # Use attention mechanism
        
        # Increase channel counts
        for layer in arch.depth_conv_layers:
            layer.out_channels = min(256, int(layer.out_channels * 1.3))
        
        arch = self._scale_to_budget(arch, target_params)
        return arch
    
    def _specialization_strategy(self, base_architecture: NetworkArchitecture) -> NetworkArchitecture:
        """Strategy for specialization stage: Task-specific optimizations"""
        
        arch = copy.deepcopy(base_architecture)
        target_params = self.config.stage_parameter_budgets[ProgressionStage.SPECIALIZATION]
        
        # Add task-specific architectural elements
        # For depth image processing, focus on spatial feature extraction
        
        # Use separable convolutions for efficiency
        for layer in arch.depth_conv_layers[::2]:  # Every other layer
            layer.block_type = ConvBlockType.SEPARABLE
        
        # Add more sophisticated activations
        for layer in arch.depth_conv_layers:
            if layer.activation == ActivationType.RELU:
                layer.activation = ActivationType.SWISH  # More expressive activation
        
        # Expand sensor processing
        if len(arch.sensor_fc_layers) < 4:
            arch.sensor_fc_layers.insert(0, arch.sensor_fc_layers[0] * 2)
        
        # More complex fusion
        if len(arch.fusion_config.hidden_sizes) < 4:
            arch.fusion_config.hidden_sizes.extend([512, 256])
        
        arch = self._scale_to_budget(arch, target_params)
        return arch
    
    def _maturation_strategy(self, base_architecture: NetworkArchitecture) -> NetworkArchitecture:
        """Strategy for maturation stage: Final optimizations"""
        
        arch = copy.deepcopy(base_architecture)
        target_params = self.config.stage_parameter_budgets[ProgressionStage.MATURATION]
        
        # Use full complexity budget
        # Add dense connections for rich feature reuse
        for layer in arch.depth_conv_layers:
            if layer.block_type in [ConvBlockType.BASIC, ConvBlockType.RESIDUAL]:
                layer.block_type = ConvBlockType.DENSE
        
        # Maximum channel counts
        for i, layer in enumerate(arch.depth_conv_layers):
            max_channels = min(512, 64 * (2 ** i))
            layer.out_channels = min(max_channels, target_params // (len(arch.depth_conv_layers) * 10))
        
        # Full fusion complexity
        arch.fusion_config.hidden_sizes = [1024, 512, 256, 128]
        arch.fusion_config.fusion_method = "attention"
        
        arch = self._scale_to_budget(arch, target_params)
        return arch
    
    def _default_strategy(self, base_architecture: NetworkArchitecture) -> NetworkArchitecture:
        """Default strategy: minimal changes"""
        return copy.deepcopy(base_architecture)
    
    def _estimate_parameters(self, architecture: NetworkArchitecture) -> int:
        """Estimate parameter count for architecture (simplified)"""
        # This would use the same logic as in the BayesianArchitectureOptimizer
        # For now, return a rough estimate
        total_params = 0
        
        # Conv layers
        prev_channels = 1  # Depth input
        for layer in architecture.depth_conv_layers:
            conv_params = prev_channels * layer.out_channels * (layer.kernel_size ** 2)
            bn_params = layer.out_channels * 2 if layer.use_batch_norm else 0
            total_params += conv_params + bn_params
            prev_channels = layer.out_channels
        
        # FC layers - rough estimate
        total_params += architecture.depth_fc_size * 512
        
        # Sensor FC
        prev_size = 16  # Sensor input size
        for size in architecture.sensor_fc_layers:
            total_params += prev_size * size
            prev_size = size
        
        # Fusion layers
        fusion_input = 512 + prev_size
        for size in architecture.fusion_config.hidden_sizes:
            total_params += fusion_input * size
            fusion_input = size
        
        # Output
        total_params += fusion_input * architecture.output_size
        
        return total_params
    
    def _scale_to_budget(self, architecture: NetworkArchitecture, target_params: int) -> NetworkArchitecture:
        """Scale architecture to fit parameter budget"""
        
        current_params = self._estimate_parameters(architecture)
        
        if current_params <= target_params:
            return architecture
        
        # Scale down by reducing channel counts
        scale_factor = (target_params / current_params) ** 0.5
        
        for layer in architecture.depth_conv_layers:
            layer.out_channels = max(8, int(layer.out_channels * scale_factor))
        
        # Scale fusion layers
        architecture.fusion_config.hidden_sizes = [
            max(32, int(size * scale_factor)) for size in architecture.fusion_config.hidden_sizes
        ]
        
        # Update estimated parameters
        architecture.estimated_parameters = self._estimate_parameters(architecture)
        
        return architecture
    
    def get_current_stage(self) -> ProgressionStage:
        """Get current progression stage"""
        return self.state.current_stage
    
    def get_current_architecture(self) -> NetworkArchitecture:
        """Get current architecture"""
        return self.current_architecture
    
    def get_progression_stats(self) -> Dict[str, Any]:
        """Get statistics about progression"""
        
        return {
            'current_stage': self.state.current_stage.value,
            'steps_in_stage': self.state.steps_in_stage,
            'best_performance_in_stage': self.state.best_performance_in_stage,
            'steps_since_improvement': self.state.steps_since_improvement,
            'architectures_tried_in_stage': len(self.state.stage_architectures),
            'total_architecture_lineage': len(self.architecture_lineage),
            'current_parameter_budget': self.config.stage_parameter_budgets[self.state.current_stage],
            'current_parameters': self.current_architecture.estimated_parameters,
            'parameter_utilization': self.current_architecture.estimated_parameters / self.config.stage_parameter_budgets[self.state.current_stage]
        }
    
    def force_stage_transition(self, training_step: int):
        """Force transition to next stage (for testing/debugging)"""
        if self.enable_debug:
            print("[ProgressiveNAS] Forcing stage transition")
        self._transition_to_next_stage(training_step)
    
    def reset_progression(self, training_step: int = 0):
        """Reset progression to initialization stage"""
        self.state = ProgressionState(
            current_stage=ProgressionStage.INITIALIZATION,
            stage_start_step=training_step,
            steps_in_stage=0,
            best_performance_in_stage=-float('inf'),
            steps_since_improvement=0,
            stage_architectures=[],
            performance_history=[]
        )
        
        self.current_architecture = copy.deepcopy(self.initial_architecture)
        self.architecture_lineage = []
        
        if self.enable_debug:
            print("[ProgressiveNAS] Reset to initialization stage")