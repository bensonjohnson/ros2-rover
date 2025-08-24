#!/usr/bin/env python3
"""
Safety Constraint Handler for Multi-Objective Bayesian Optimization
Ensures that optimization never suggests configurations that violate safety constraints
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import warnings

try:
    from botorch.models import SingleTaskGP, ModelListGP
    from botorch.fit import fit_gpytorch_mll
    from botorch.acquisition.multi_objective import qExpectedHypervolumeImprovement
    from botorch.acquisition.utils import get_acquisition_function
    from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
    BOTORCH_AVAILABLE = True
except ImportError:
    BOTORCH_AVAILABLE = False

class ConstraintType(Enum):
    """Types of safety constraints"""
    HARD_BOUND = "hard_bound"           # Parameter must be within bounds (e.g., max_speed <= 0.3)
    SOFT_BOUND = "soft_bound"           # Parameter should be within bounds but can be violated with penalty
    CONDITIONAL = "conditional"         # Constraint depends on other parameters
    PERFORMANCE = "performance"         # Constraint on performance metrics (e.g., collision_rate <= 0.05)

@dataclass
class SafetyConstraint:
    """Definition of a safety constraint"""
    name: str
    constraint_type: ConstraintType
    constraint_function: Callable[[Dict[str, float]], float]  # Returns constraint value (>= 0 is feasible)
    violation_penalty: float = 1000.0                        # Penalty for constraint violation
    description: str = ""

class SafetyConstraintHandler:
    """
    Handles safety constraints for multi-objective Bayesian optimization.
    
    Ensures that suggested parameter configurations never violate critical safety bounds
    like maximum speeds, minimum safety distances, maximum collision rates, etc.
    """
    
    def __init__(self, enable_debug: bool = False):
        self.enable_debug = enable_debug
        self.constraints: List[SafetyConstraint] = []
        
        # Constraint violation tracking
        self.violation_history: List[Dict[str, float]] = []
        self.feasible_parameter_history: List[Dict[str, float]] = []
        
        # GP models for constraint prediction
        self.constraint_models: Optional[ModelListGP] = None
        
        # Default rover safety constraints
        self._setup_default_rover_constraints()
        
        if self.enable_debug:
            print(f"[SafetyConstraints] Initialized with {len(self.constraints)} default constraints")
    
    def _setup_default_rover_constraints(self):
        """Setup default safety constraints for rover operations"""
        
        # 1. Maximum speed constraint (hard physical limit)
        def max_speed_constraint(params: Dict[str, float]) -> float:
            max_allowed_speed = 0.5  # 0.5 m/s maximum for safety
            current_max_speed = params.get('max_speed', 0.15)
            return max_allowed_speed - current_max_speed  # >= 0 is feasible
        
        self.add_constraint(SafetyConstraint(
            name="max_speed_limit",
            constraint_type=ConstraintType.HARD_BOUND,
            constraint_function=max_speed_constraint,
            violation_penalty=10000.0,
            description="Maximum rover speed must not exceed 0.5 m/s"
        ))
        
        # 2. Minimum safety distance constraint
        def min_safety_distance_constraint(params: Dict[str, float]) -> float:
            min_allowed_distance = 0.05  # 5cm minimum safety margin
            current_safety_distance = params.get('safety_distance', 0.2)
            return current_safety_distance - min_allowed_distance  # >= 0 is feasible
        
        self.add_constraint(SafetyConstraint(
            name="min_safety_distance",
            constraint_type=ConstraintType.HARD_BOUND,
            constraint_function=min_safety_distance_constraint,
            violation_penalty=5000.0,
            description="Safety distance must be at least 5cm"
        ))
        
        # 3. Learning rate bounds (prevent training instability)
        def learning_rate_bounds_constraint(params: Dict[str, float]) -> float:
            min_lr = 1e-6
            max_lr = 0.1
            current_lr = params.get('learning_rate', 0.001)
            # Must be within bounds
            lower_bound = current_lr - min_lr
            upper_bound = max_lr - current_lr
            return min(lower_bound, upper_bound)  # Both must be >= 0
        
        self.add_constraint(SafetyConstraint(
            name="learning_rate_bounds",
            constraint_type=ConstraintType.HARD_BOUND,
            constraint_function=learning_rate_bounds_constraint,
            violation_penalty=2000.0,
            description="Learning rate must be between 1e-6 and 0.1"
        ))
        
        # 4. Batch size constraints (prevent memory issues)
        def batch_size_constraint(params: Dict[str, float]) -> float:
            min_batch_size = 4
            max_batch_size = 256
            current_batch_size = params.get('batch_size', 32)
            # Must be within bounds and power of 2
            if current_batch_size < min_batch_size:
                return min_batch_size - current_batch_size
            elif current_batch_size > max_batch_size:
                return max_batch_size - current_batch_size
            else:
                return 1.0  # Feasible
        
        self.add_constraint(SafetyConstraint(
            name="batch_size_bounds",
            constraint_type=ConstraintType.SOFT_BOUND,
            constraint_function=batch_size_constraint,
            violation_penalty=500.0,
            description="Batch size should be between 4 and 256"
        ))
        
        # 5. Exploration time constraint (prevent excessive runs)
        def exploration_time_constraint(params: Dict[str, float]) -> float:
            max_exploration_time = 3600  # 1 hour maximum
            current_time = params.get('exploration_time', 300)
            return max_exploration_time - current_time
        
        self.add_constraint(SafetyConstraint(
            name="max_exploration_time",
            constraint_type=ConstraintType.SOFT_BOUND,
            constraint_function=exploration_time_constraint,
            violation_penalty=1000.0,
            description="Exploration time should not exceed 1 hour"
        ))
        
        # 6. Conditional constraint: High speed requires larger safety distance
        def speed_safety_conditional_constraint(params: Dict[str, float]) -> float:
            max_speed = params.get('max_speed', 0.15)
            safety_distance = params.get('safety_distance', 0.2)
            
            # Required safety distance scales with speed
            required_safety_distance = 0.1 + (max_speed * 0.5)  # Linear scaling
            
            return safety_distance - required_safety_distance
        
        self.add_constraint(SafetyConstraint(
            name="speed_safety_coupling",
            constraint_type=ConstraintType.CONDITIONAL,
            constraint_function=speed_safety_conditional_constraint,
            violation_penalty=3000.0,
            description="Higher speeds require proportionally larger safety distances"
        ))
    
    def add_constraint(self, constraint: SafetyConstraint):
        """Add a safety constraint to the handler"""
        self.constraints.append(constraint)
        
        if self.enable_debug:
            print(f"[SafetyConstraints] Added constraint: {constraint.name}")
    
    def evaluate_constraints(self, parameters: Dict[str, float]) -> Dict[str, float]:
        """
        Evaluate all constraints for given parameters.
        
        Returns:
            Dict mapping constraint names to constraint values (>= 0 is feasible)
        """
        constraint_values = {}
        
        for constraint in self.constraints:
            try:
                constraint_value = constraint.constraint_function(parameters)
                constraint_values[constraint.name] = constraint_value
            except Exception as e:
                if self.enable_debug:
                    print(f"[SafetyConstraints] Error evaluating {constraint.name}: {e}")
                # Assume constraint violated if evaluation fails
                constraint_values[constraint.name] = -1.0
        
        return constraint_values
    
    def is_feasible(self, parameters: Dict[str, float]) -> bool:
        """Check if parameters satisfy all hard constraints"""
        constraint_values = self.evaluate_constraints(parameters)
        
        for constraint in self.constraints:
            if constraint.constraint_type == ConstraintType.HARD_BOUND:
                constraint_value = constraint_values.get(constraint.name, -1.0)
                if constraint_value < 0:
                    if self.enable_debug:
                        print(f"[SafetyConstraints] Hard constraint violated: {constraint.name} = {constraint_value}")
                    return False
        
        return True
    
    def calculate_constraint_penalty(self, parameters: Dict[str, float]) -> float:
        """Calculate total penalty for constraint violations"""
        constraint_values = self.evaluate_constraints(parameters)
        total_penalty = 0.0
        
        for constraint in self.constraints:
            constraint_value = constraint_values.get(constraint.name, -1.0)
            
            if constraint_value < 0:  # Constraint violated
                violation_magnitude = abs(constraint_value)
                penalty = constraint.violation_penalty * violation_magnitude
                total_penalty += penalty
                
                if self.enable_debug:
                    print(f"[SafetyConstraints] Constraint {constraint.name} violated: "
                          f"value={constraint_value:.4f}, penalty={penalty:.2f}")
        
        return total_penalty
    
    def repair_parameters(self, parameters: Dict[str, float]) -> Dict[str, float]:
        """
        Attempt to repair parameter configuration to satisfy hard constraints.
        
        This is a simple repair mechanism that adjusts parameters to barely satisfy constraints.
        """
        repaired_params = parameters.copy()
        
        # Iterative repair process
        max_iterations = 10
        iteration = 0
        
        while iteration < max_iterations and not self.is_feasible(repaired_params):
            constraint_values = self.evaluate_constraints(repaired_params)
            
            for constraint in self.constraints:
                if constraint.constraint_type != ConstraintType.HARD_BOUND:
                    continue
                
                constraint_value = constraint_values.get(constraint.name, 0.0)
                
                if constraint_value < 0:  # Violated
                    # Apply parameter-specific repair strategies
                    if constraint.name == "max_speed_limit":
                        repaired_params['max_speed'] = min(repaired_params.get('max_speed', 0.15), 0.4)
                    
                    elif constraint.name == "min_safety_distance":
                        repaired_params['safety_distance'] = max(repaired_params.get('safety_distance', 0.2), 0.1)
                    
                    elif constraint.name == "learning_rate_bounds":
                        lr = repaired_params.get('learning_rate', 0.001)
                        repaired_params['learning_rate'] = np.clip(lr, 1e-6, 0.05)
                    
                    elif constraint.name == "batch_size_bounds":
                        batch_size = repaired_params.get('batch_size', 32)
                        # Round to nearest valid power of 2 within bounds
                        valid_batch_sizes = [4, 8, 16, 32, 64, 128, 256]
                        repaired_params['batch_size'] = min(valid_batch_sizes, 
                                                          key=lambda x: abs(x - batch_size))
                    
                    elif constraint.name == "speed_safety_coupling":
                        # Increase safety distance to match speed
                        max_speed = repaired_params.get('max_speed', 0.15)
                        required_safety = 0.1 + (max_speed * 0.5)
                        repaired_params['safety_distance'] = max(repaired_params.get('safety_distance', 0.2), 
                                                               required_safety)
            
            iteration += 1
        
        if self.enable_debug and iteration > 0:
            print(f"[SafetyConstraints] Repaired parameters in {iteration} iterations")
            if not self.is_feasible(repaired_params):
                print(f"[SafetyConstraints] Warning: Could not fully repair parameters")
        
        return repaired_params
    
    def filter_candidates(self, candidate_list: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """Filter candidate parameters to only include feasible ones"""
        
        feasible_candidates = []
        
        for candidate in candidate_list:
            if self.is_feasible(candidate):
                feasible_candidates.append(candidate)
            else:
                # Try to repair the candidate
                repaired_candidate = self.repair_parameters(candidate)
                if self.is_feasible(repaired_candidate):
                    feasible_candidates.append(repaired_candidate)
                elif self.enable_debug:
                    print(f"[SafetyConstraints] Rejected infeasible candidate: {candidate}")
        
        if self.enable_debug:
            print(f"[SafetyConstraints] Filtered {len(candidate_list)} candidates to {len(feasible_candidates)} feasible ones")
        
        return feasible_candidates
    
    def update_constraint_models(self, 
                               parameter_history: List[Dict[str, float]], 
                               constraint_evaluation_history: List[Dict[str, float]]):
        """Update GP models for constraint prediction"""
        
        if not BOTORCH_AVAILABLE or len(parameter_history) < 5:
            return
        
        try:
            # Convert parameter history to tensor
            param_names = list(parameter_history[0].keys()) if parameter_history else []
            if not param_names:
                return
            
            # Normalize parameters to [0,1] (would need bounds from elsewhere)
            X_list = []
            for params in parameter_history:
                param_row = [params.get(name, 0.0) for name in param_names]
                X_list.append(param_row)
            
            X = torch.tensor(X_list, dtype=torch.float64)
            
            # Create constraint value tensors
            constraint_models = []
            
            for constraint in self.constraints:
                constraint_name = constraint.name
                y_list = []
                
                for constraint_eval in constraint_evaluation_history:
                    constraint_value = constraint_eval.get(constraint_name, 0.0)
                    y_list.append(constraint_value)
                
                if len(y_list) == len(parameter_history):
                    Y = torch.tensor(y_list, dtype=torch.float64).unsqueeze(-1)
                    
                    # Create GP model for this constraint
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        constraint_model = SingleTaskGP(X, Y)
                        mll = ExactMarginalLogLikelihood(constraint_model.likelihood, constraint_model)
                        fit_gpytorch_mll(mll)
                        constraint_models.append(constraint_model)
            
            if constraint_models:
                self.constraint_models = ModelListGP(*constraint_models)
                
                if self.enable_debug:
                    print(f"[SafetyConstraints] Updated constraint models for {len(constraint_models)} constraints")
        
        except Exception as e:
            if self.enable_debug:
                print(f"[SafetyConstraints] Failed to update constraint models: {e}")
    
    def predict_constraint_satisfaction(self, parameters: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
        """
        Predict constraint satisfaction using GP models.
        
        Returns:
            Dict mapping constraint names to (mean, std) predictions
        """
        
        if self.constraint_models is None:
            return {}
        
        predictions = {}
        
        try:
            # Convert parameters to normalized tensor (would need proper normalization)
            param_list = [parameters.get(name, 0.0) for name in parameters.keys()]
            X_test = torch.tensor([param_list], dtype=torch.float64)
            
            with torch.no_grad():
                for i, constraint in enumerate(self.constraints):
                    if i < len(self.constraint_models.models):
                        model = self.constraint_models.models[i]
                        posterior = model.posterior(X_test)
                        mean = posterior.mean.item()
                        variance = posterior.variance.item()
                        std = np.sqrt(variance)
                        
                        predictions[constraint.name] = (mean, std)
        
        except Exception as e:
            if self.enable_debug:
                print(f"[SafetyConstraints] Failed to predict constraints: {e}")
        
        return predictions
    
    def get_constraint_summary(self) -> Dict[str, any]:
        """Get summary of all constraints and their current status"""
        
        summary = {
            'total_constraints': len(self.constraints),
            'constraint_types': {},
            'constraints': []
        }
        
        # Count constraint types
        for constraint in self.constraints:
            constraint_type = constraint.constraint_type.value
            summary['constraint_types'][constraint_type] = summary['constraint_types'].get(constraint_type, 0) + 1
        
        # Constraint details
        for constraint in self.constraints:
            constraint_info = {
                'name': constraint.name,
                'type': constraint.constraint_type.value,
                'penalty': constraint.violation_penalty,
                'description': constraint.description
            }
            summary['constraints'].append(constraint_info)
        
        return summary
    
    def create_safety_aware_acquisition_function(self, base_acquisition_function):
        """
        Create a safety-aware acquisition function that incorporates constraints.
        
        This would modify the acquisition function to penalize areas of parameter space
        that are likely to violate constraints.
        """
        
        if self.constraint_models is None:
            return base_acquisition_function
        
        # This is a placeholder for constraint-aware acquisition function
        # In practice, would use BoTorch's constraint handling capabilities
        # like qExpectedHypervolumeImprovement with constraint models
        
        # For now, return the base acquisition function
        # TODO: Implement proper constrained acquisition function
        return base_acquisition_function