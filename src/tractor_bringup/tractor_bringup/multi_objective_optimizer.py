#!/usr/bin/env python3
"""
Multi-Objective Bayesian Optimization with Pareto Frontier
Uses BoTorch's multi-objective capabilities to simultaneously optimize performance, safety, efficiency, and robustness
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings

try:
    import botorch
    from botorch.models import ModelListGP, SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from botorch.acquisition.multi_objective import qExpectedHypervolumeImprovement, qNoisyExpectedHypervolumeImprovement
    from botorch.optim import optimize_acqf
    from botorch.utils.multi_objective import is_non_dominated
    from botorch.utils.transforms import unnormalize, normalize
    from gpytorch.mlls import SumMarginalLogLikelihood, ExactMarginalLogLikelihood
    BOTORCH_AVAILABLE = True
    print("BoTorch multi-objective capabilities successfully imported")
except ImportError as e:
    BOTORCH_AVAILABLE = False
    print(f"BoTorch multi-objective not available: {e}")

@dataclass
class MultiObjectivePoint:
    """Container for a multi-objective evaluation point"""
    parameters: Dict[str, float]
    objectives: Dict[str, float]  # performance, safety, efficiency, robustness
    constraints: Dict[str, float]  # safety constraints
    timestamp: float
    evaluation_id: int

@dataclass
class ParetoFrontierSolution:
    """Container for a Pareto-optimal solution"""
    parameters: Dict[str, float]
    objectives: Dict[str, float]
    hypervolume_contribution: float
    dominates_count: int  # How many other solutions this dominates

class MultiObjectiveBayesianOptimizer:
    """
    Multi-objective Bayesian optimizer using Pareto frontiers.
    
    Simultaneously optimizes multiple competing objectives:
    - Performance: Exploration efficiency, reward accumulation
    - Safety: Collision avoidance, safety margins
    - Efficiency: Energy consumption, movement efficiency  
    - Robustness: Consistency, recovery from failures
    
    Uses qNEHVI (q-Noisy Expected Hypervolume Improvement) for acquisition.
    """
    
    def __init__(self,
                 parameter_bounds: Dict[str, Tuple[float, float]],
                 objective_names: List[str] = None,
                 constraint_names: List[str] = None,
                 reference_point: Optional[List[float]] = None,
                 enable_debug: bool = False):
        
        if not BOTORCH_AVAILABLE:
            raise ImportError("BoTorch is required for multi-objective optimization. Install with: pip install botorch")
        
        self.enable_debug = enable_debug
        self.parameter_bounds = parameter_bounds
        self.param_names = list(parameter_bounds.keys())
        self.n_params = len(self.param_names)
        
        # Objectives to optimize (all maximization - we'll negate minimization objectives)
        self.objective_names = objective_names or ['performance', 'safety', 'efficiency', 'robustness']
        self.n_objectives = len(self.objective_names)
        
        # Safety constraints (inequality constraints c(x) >= 0)
        self.constraint_names = constraint_names or ['min_safety_distance', 'max_collision_rate']
        self.n_constraints = len(self.constraint_names)
        
        # Reference point for hypervolume calculation (slightly below worst expected values)
        self.reference_point = torch.tensor(reference_point or [0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        
        # Convert parameter bounds to torch tensors for BoTorch
        bounds_array = np.array([parameter_bounds[name] for name in self.param_names])
        self.bounds = torch.tensor(bounds_array.T, dtype=torch.float64)
        
        # Storage for evaluations
        self.evaluation_history: List[MultiObjectivePoint] = []
        self.evaluation_counter = 0
        
        # Pareto frontier tracking
        self.pareto_frontier: List[ParetoFrontierSolution] = []
        self.hypervolume_history: List[float] = []
        
        # BoTorch model components
        self.gp_models: Optional[ModelListGP] = None
        self.constraint_models: Optional[ModelListGP] = None
        self.mll = None
        
        # Acquisition function parameters
        self.acquisition_restarts = 20
        self.acquisition_raw_samples = 1024
        self.batch_size = 4  # Number of candidates to suggest simultaneously
        
        if self.enable_debug:
            print(f"[MultiObjective] Initialized with {self.n_params} parameters, {self.n_objectives} objectives")
            print(f"[MultiObjective] Parameter bounds: {parameter_bounds}")
            print(f"[MultiObjective] Reference point: {self.reference_point}")
    
    def add_evaluation(self, 
                      parameters: Dict[str, float],
                      objectives: Dict[str, float],
                      constraints: Optional[Dict[str, float]] = None):
        """Add a multi-objective evaluation to the history"""
        
        # Validate inputs
        if set(parameters.keys()) != set(self.param_names):
            raise ValueError(f"Parameters must match expected names: {self.param_names}")
        if set(objectives.keys()) != set(self.objective_names):
            raise ValueError(f"Objectives must match expected names: {self.objective_names}")
        
        constraints = constraints or {}
        
        # Create evaluation point
        evaluation = MultiObjectivePoint(
            parameters=parameters.copy(),
            objectives=objectives.copy(),
            constraints=constraints.copy(),
            timestamp=torch.tensor(0.0).item(),  # Would use time.time() in real implementation
            evaluation_id=self.evaluation_counter
        )
        
        self.evaluation_history.append(evaluation)
        self.evaluation_counter += 1
        
        # Update Pareto frontier
        self._update_pareto_frontier()
        
        # Update GP models
        self._update_models()
        
        if self.enable_debug:
            print(f"[MultiObjective] Added evaluation {self.evaluation_counter}")
            print(f"[MultiObjective] Objectives: {objectives}")
            print(f"[MultiObjective] Current Pareto frontier size: {len(self.pareto_frontier)}")
    
    def suggest_candidates(self, n_candidates: int = None) -> List[Dict[str, float]]:
        """
        Suggest next parameter candidates using multi-objective Bayesian optimization.
        
        Returns a list of parameter dictionaries representing diverse Pareto-optimal candidates.
        """
        
        n_candidates = n_candidates or self.batch_size
        
        # Need at least a few evaluations to build GP models
        if len(self.evaluation_history) < 3:
            return self._random_exploration(n_candidates)
        
        if self.gp_models is None:
            return self._random_exploration(n_candidates)
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Get current Pareto frontier for reference point refinement
                pareto_objectives = self._get_pareto_objective_values()
                
                # Create acquisition function (qNEHVI - q-Noisy Expected Hypervolume Improvement)
                acquisition_func = qNoisyExpectedHypervolumeImprovement(
                    model=self.gp_models,
                    ref_point=self.reference_point,
                    X_baseline=self._get_parameter_tensor(),
                    prune_baseline=True,
                    sampler=None  # Use default sampler
                )
                
                # Add constraint handling if we have constraint models
                if self.constraint_models is not None:
                    # Would implement constraint handling here
                    pass
                
                # Optimize acquisition function to find next candidates
                candidates_tensor, _ = optimize_acqf(
                    acq_function=acquisition_func,
                    bounds=self.bounds,
                    q=n_candidates,
                    num_restarts=self.acquisition_restarts,
                    raw_samples=self.acquisition_raw_samples,
                    options={"batch_limit": 5, "maxiter": 200}
                )
                
                # Convert tensor candidates back to parameter dictionaries
                candidates = []
                for i in range(candidates_tensor.shape[0]):
                    candidate_dict = {}
                    for j, param_name in enumerate(self.param_names):
                        value = candidates_tensor[i, j].item()
                        
                        # Apply any parameter-specific post-processing
                        if 'batch_size' in param_name:
                            # Round batch sizes to powers of 2
                            value = max(8, 2 ** round(np.log2(max(8, value))))
                        elif 'frequency' in param_name or 'steps' in param_name:
                            # Round frequency/step parameters to integers
                            value = int(round(value))
                        
                        candidate_dict[param_name] = value
                    
                    candidates.append(candidate_dict)
                
                if self.enable_debug:
                    print(f"[MultiObjective] Suggested {len(candidates)} candidates using qNEHVI")
                    for i, candidate in enumerate(candidates):
                        print(f"[MultiObjective] Candidate {i+1}: {candidate}")
                
                return candidates
                
        except Exception as e:
            if self.enable_debug:
                print(f"[MultiObjective] Acquisition optimization failed: {e}, using random exploration")
            return self._random_exploration(n_candidates)
    
    def _random_exploration(self, n_candidates: int) -> List[Dict[str, float]]:
        """Generate random parameter candidates for exploration"""
        
        candidates = []
        for _ in range(n_candidates):
            candidate = {}
            for param_name in self.param_names:
                min_val, max_val = self.parameter_bounds[param_name]
                
                if 'batch_size' in param_name:
                    # Random power of 2 for batch sizes
                    log_min, log_max = np.log2(min_val), np.log2(max_val)
                    log_val = np.random.uniform(log_min, log_max)
                    value = max(8, 2 ** round(log_val))
                elif 'frequency' in param_name or 'steps' in param_name:
                    # Random integer for frequency/step parameters
                    value = int(np.random.uniform(min_val, max_val))
                elif 'learning_rate' in param_name or 'weight_decay' in param_name:
                    # Log scale for learning rates
                    log_min, log_max = np.log10(min_val), np.log10(max_val)
                    log_val = np.random.uniform(log_min, log_max)
                    value = 10 ** log_val
                else:
                    # Linear scale for other parameters
                    value = np.random.uniform(min_val, max_val)
                
                candidate[param_name] = value
            
            candidates.append(candidate)
        
        if self.enable_debug:
            print(f"[MultiObjective] Generated {n_candidates} random exploration candidates")
        
        return candidates
    
    def _update_pareto_frontier(self):
        """Update the Pareto frontier with current evaluations"""
        
        if len(self.evaluation_history) < 2:
            return
        
        # Get objective values as tensor
        objective_values = []
        for eval_point in self.evaluation_history:
            obj_vals = [eval_point.objectives[name] for name in self.objective_names]
            objective_values.append(obj_vals)
        
        objective_tensor = torch.tensor(objective_values, dtype=torch.float64)
        
        # Find non-dominated points (Pareto frontier)
        is_pareto = is_non_dominated(objective_tensor)
        
        # Update Pareto frontier solutions
        self.pareto_frontier = []
        for i, is_pareto_optimal in enumerate(is_pareto):
            if is_pareto_optimal:
                eval_point = self.evaluation_history[i]
                
                # Calculate hypervolume contribution (simplified)
                hv_contribution = self._calculate_hypervolume_contribution(eval_point.objectives)
                
                # Count how many solutions this dominates
                dominates_count = self._count_dominated_solutions(i, objective_tensor)
                
                pareto_solution = ParetoFrontierSolution(
                    parameters=eval_point.parameters.copy(),
                    objectives=eval_point.objectives.copy(),
                    hypervolume_contribution=hv_contribution,
                    dominates_count=dominates_count
                )
                
                self.pareto_frontier.append(pareto_solution)
        
        # Calculate current hypervolume
        if len(self.pareto_frontier) > 0:
            current_hv = self._calculate_hypervolume()
            self.hypervolume_history.append(current_hv)
        
        if self.enable_debug:
            print(f"[MultiObjective] Updated Pareto frontier: {len(self.pareto_frontier)} solutions")
    
    def _update_models(self):
        """Update GP models for objectives and constraints"""
        
        if len(self.evaluation_history) < 3:
            return
        
        try:
            # Prepare training data
            X = self._get_parameter_tensor()
            Y_objectives = self._get_objective_tensor()
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Create individual GP models for each objective
                objective_models = []
                for i in range(self.n_objectives):
                    y_i = Y_objectives[:, i:i+1]  # Keep as 2D tensor
                    model_i = SingleTaskGP(X, y_i)
                    objective_models.append(model_i)
                
                # Combine into ModelListGP
                self.gp_models = ModelListGP(*objective_models)
                
                # Fit models
                mll = SumMarginalLogLikelihood(self.gp_models.likelihood, self.gp_models)
                fit_gpytorch_mll(mll)
                self.mll = mll
                
                # TODO: Create constraint models if we have constraint data
                # This would be similar but for constraint values
                
                if self.enable_debug:
                    print(f"[MultiObjective] Updated GP models for {self.n_objectives} objectives")
                    
        except Exception as e:
            if self.enable_debug:
                print(f"[MultiObjective] Failed to update GP models: {e}")
            self.gp_models = None
    
    def _get_parameter_tensor(self) -> torch.Tensor:
        """Get parameter values as normalized tensor"""
        param_values = []
        for eval_point in self.evaluation_history:
            param_row = []
            for param_name in self.param_names:
                value = eval_point.parameters[param_name]
                min_val, max_val = self.parameter_bounds[param_name]
                # Normalize to [0, 1]
                normalized = (value - min_val) / (max_val - min_val)
                param_row.append(normalized)
            param_values.append(param_row)
        
        return torch.tensor(param_values, dtype=torch.float64)
    
    def _get_objective_tensor(self) -> torch.Tensor:
        """Get objective values as tensor"""
        objective_values = []
        for eval_point in self.evaluation_history:
            obj_row = [eval_point.objectives[name] for name in self.objective_names]
            objective_values.append(obj_row)
        
        return torch.tensor(objective_values, dtype=torch.float64)
    
    def _get_pareto_objective_values(self) -> torch.Tensor:
        """Get objective values for current Pareto frontier"""
        if not self.pareto_frontier:
            return torch.empty(0, self.n_objectives, dtype=torch.float64)
        
        pareto_objectives = []
        for solution in self.pareto_frontier:
            obj_row = [solution.objectives[name] for name in self.objective_names]
            pareto_objectives.append(obj_row)
        
        return torch.tensor(pareto_objectives, dtype=torch.float64)
    
    def _calculate_hypervolume_contribution(self, objectives: Dict[str, float]) -> float:
        """Calculate hypervolume contribution of a point (simplified)"""
        
        # Simplified hypervolume contribution calculation
        # In practice, would use more sophisticated hypervolume algorithms
        obj_values = [objectives[name] for name in self.objective_names]
        ref_values = self.reference_point.tolist()
        
        # Volume is product of improvements over reference point
        volume = 1.0
        for obj_val, ref_val in zip(obj_values, ref_values):
            improvement = max(0, obj_val - ref_val)
            volume *= improvement
        
        return volume
    
    def _count_dominated_solutions(self, solution_idx: int, objective_tensor: torch.Tensor) -> int:
        """Count how many solutions this one dominates"""
        
        target_objectives = objective_tensor[solution_idx]
        dominates_count = 0
        
        for i, other_objectives in enumerate(objective_tensor):
            if i == solution_idx:
                continue
                
            # Check if target dominates other (target >= other in all objectives, > in at least one)
            dominates = True
            strictly_better_in_one = False
            
            for j in range(len(target_objectives)):
                if target_objectives[j] < other_objectives[j]:
                    dominates = False
                    break
                if target_objectives[j] > other_objectives[j]:
                    strictly_better_in_one = True
            
            if dominates and strictly_better_in_one:
                dominates_count += 1
        
        return dominates_count
    
    def _calculate_hypervolume(self) -> float:
        """Calculate hypervolume of current Pareto frontier (simplified)"""
        
        if not self.pareto_frontier:
            return 0.0
        
        # Simplified hypervolume calculation
        # Sum of individual contributions (not accurate for overlapping regions)
        total_hv = sum(solution.hypervolume_contribution for solution in self.pareto_frontier)
        
        return total_hv
    
    def get_pareto_frontier(self) -> List[ParetoFrontierSolution]:
        """Get current Pareto frontier solutions"""
        return self.pareto_frontier.copy()
    
    def get_best_solution_for_objective(self, objective_name: str) -> Optional[ParetoFrontierSolution]:
        """Get the Pareto solution that excels most in a specific objective"""
        
        if not self.pareto_frontier or objective_name not in self.objective_names:
            return None
        
        best_solution = max(self.pareto_frontier, 
                          key=lambda sol: sol.objectives[objective_name])
        
        return best_solution
    
    def get_most_balanced_solution(self) -> Optional[ParetoFrontierSolution]:
        """Get the Pareto solution with most balanced objective values"""
        
        if not self.pareto_frontier:
            return None
        
        # Find solution with minimum variance across normalized objectives
        best_solution = None
        min_variance = float('inf')
        
        for solution in self.pareto_frontier:
            obj_values = [solution.objectives[name] for name in self.objective_names]
            # Normalize to [0,1] based on frontier range
            if len(obj_values) > 1:
                variance = np.var(obj_values)
                if variance < min_variance:
                    min_variance = variance
                    best_solution = solution
        
        return best_solution or self.pareto_frontier[0]
    
    def get_optimization_statistics(self) -> Dict[str, Union[float, int, List]]:
        """Get comprehensive optimization statistics"""
        
        stats = {
            'n_evaluations': len(self.evaluation_history),
            'pareto_frontier_size': len(self.pareto_frontier),
            'current_hypervolume': self.hypervolume_history[-1] if self.hypervolume_history else 0.0,
            'hypervolume_improvement': 0.0,
            'objective_ranges': {},
            'parameter_diversity': {}
        }
        
        # Calculate hypervolume improvement
        if len(self.hypervolume_history) >= 2:
            stats['hypervolume_improvement'] = self.hypervolume_history[-1] - self.hypervolume_history[0]
        
        # Calculate objective ranges
        if self.evaluation_history:
            for obj_name in self.objective_names:
                obj_values = [eval_pt.objectives[obj_name] for eval_pt in self.evaluation_history]
                stats['objective_ranges'][obj_name] = {
                    'min': min(obj_values),
                    'max': max(obj_values),
                    'mean': np.mean(obj_values),
                    'std': np.std(obj_values)
                }
        
        # Calculate parameter diversity (coefficient of variation)
        if self.evaluation_history:
            for param_name in self.param_names:
                param_values = [eval_pt.parameters[param_name] for eval_pt in self.evaluation_history]
                mean_val = np.mean(param_values)
                std_val = np.std(param_values)
                cv = std_val / mean_val if mean_val != 0 else 0
                stats['parameter_diversity'][param_name] = cv
        
        return stats
    
    def save_state(self, filepath: str):
        """Save optimization state to file"""
        try:
            state = {
                'parameter_bounds': self.parameter_bounds,
                'objective_names': self.objective_names,
                'constraint_names': self.constraint_names,
                'reference_point': self.reference_point.tolist(),
                'evaluation_history': [
                    {
                        'parameters': eval_pt.parameters,
                        'objectives': eval_pt.objectives,
                        'constraints': eval_pt.constraints,
                        'evaluation_id': eval_pt.evaluation_id
                    } for eval_pt in self.evaluation_history
                ],
                'pareto_frontier': [
                    {
                        'parameters': sol.parameters,
                        'objectives': sol.objectives,
                        'hypervolume_contribution': sol.hypervolume_contribution,
                        'dominates_count': sol.dominates_count
                    } for sol in self.pareto_frontier
                ],
                'hypervolume_history': self.hypervolume_history
            }
            
            torch.save(state, filepath)
            
            if self.enable_debug:
                print(f"[MultiObjective] Saved optimization state to {filepath}")
                
        except Exception as e:
            if self.enable_debug:
                print(f"[MultiObjective] Failed to save state: {e}")
    
    def load_state(self, filepath: str):
        """Load optimization state from file"""
        try:
            state = torch.load(filepath, map_location='cpu')
            
            # Restore evaluation history
            self.evaluation_history = []
            for eval_data in state['evaluation_history']:
                eval_pt = MultiObjectivePoint(
                    parameters=eval_data['parameters'],
                    objectives=eval_data['objectives'],
                    constraints=eval_data['constraints'],
                    timestamp=0.0,
                    evaluation_id=eval_data['evaluation_id']
                )
                self.evaluation_history.append(eval_pt)
            
            # Restore Pareto frontier
            self.pareto_frontier = []
            for sol_data in state['pareto_frontier']:
                solution = ParetoFrontierSolution(
                    parameters=sol_data['parameters'],
                    objectives=sol_data['objectives'],
                    hypervolume_contribution=sol_data['hypervolume_contribution'],
                    dominates_count=sol_data['dominates_count']
                )
                self.pareto_frontier.append(solution)
            
            self.hypervolume_history = state['hypervolume_history']
            self.evaluation_counter = len(self.evaluation_history)
            
            # Rebuild models if we have data
            if len(self.evaluation_history) >= 3:
                self._update_models()
            
            if self.enable_debug:
                print(f"[MultiObjective] Loaded optimization state from {filepath}")
                print(f"[MultiObjective] Loaded {len(self.evaluation_history)} evaluations")
                
        except Exception as e:
            if self.enable_debug:
                print(f"[MultiObjective] Failed to load state: {e}")