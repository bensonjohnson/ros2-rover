#!/usr/bin/env python3
"""
Optimization Monitoring and Logging System
Provides comprehensive monitoring, logging, and visualization for all Bayesian optimization components
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.animation import FuncAnimation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib not available - visualization features disabled")

class OptimizationMonitor:
    """
    Comprehensive monitoring system for Bayesian optimization processes.
    
    Tracks and logs all optimization components:
    - ES hyperparameter optimization
    - Training parameter optimization  
    - Reward parameter optimization
    - Multi-metric fitness evaluation
    """
    
    def __init__(self, 
                 log_dir: str = "logs/optimization",
                 enable_visualization: bool = False,
                 enable_debug: bool = False):
        
        self.log_dir = log_dir
        self.enable_visualization = enable_visualization and MATPLOTLIB_AVAILABLE
        self.enable_debug = enable_debug
        
        # Create log directory structure
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, "es_optimization"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "training_optimization"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "reward_optimization"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "multi_metric"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "plots"), exist_ok=True)
        
        # Initialize logging data structures
        self.optimization_logs = defaultdict(list)
        self.performance_logs = defaultdict(list)
        self.parameter_logs = defaultdict(list)
        
        # Real-time monitoring
        self.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_start_time = time.time()
        self.log_update_interval = 30  # seconds
        self.last_log_update = 0
        
        # Metric tracking
        self.metric_history = {
            'es_fitness': deque(maxlen=1000),
            'training_fitness': deque(maxlen=1000),
            'reward_fitness': deque(maxlen=1000),
            'overall_fitness': deque(maxlen=1000),
            'performance_score': deque(maxlen=1000),
            'safety_score': deque(maxlen=1000),
            'efficiency_score': deque(maxlen=1000),
            'robustness_score': deque(maxlen=1000)
        }
        
        # Visualization setup
        if self.enable_visualization:
            self.setup_visualization()
        
        # Initialize session log file
        self.session_log_file = os.path.join(log_dir, f"session_{self.current_session_id}.jsonl")
        
        if self.enable_debug:
            print(f"[OptimizationMonitor] Initialized with session ID: {self.current_session_id}")
            print(f"[OptimizationMonitor] Log directory: {log_dir}")
    
    def log_es_optimization(self, 
                           generation: int,
                           fitness_score: float,
                           best_params: Dict[str, Any],
                           population_stats: Dict[str, float]):
        """Log ES optimization step"""
        
        timestamp = time.time()
        
        log_entry = {
            'timestamp': timestamp,
            'session_id': self.current_session_id,
            'optimization_type': 'es',
            'generation': generation,
            'fitness_score': fitness_score,
            'best_params': best_params,
            'population_stats': population_stats
        }
        
        self.optimization_logs['es'].append(log_entry)
        self.metric_history['es_fitness'].append(fitness_score)
        
        # Write to file periodically
        if timestamp - self.last_log_update > self.log_update_interval:
            self._write_logs_to_file()
        
        if self.enable_debug:
            print(f"[OptMon] ES Gen {generation}: Fitness={fitness_score:.4f}, "
                  f"Best Sigma={best_params.get('sigma', 0):.4f}")
    
    def log_training_optimization(self,
                                step: int,
                                fitness_score: float,
                                best_params: Dict[str, Any],
                                training_stats: Dict[str, float]):
        """Log training parameter optimization step"""
        
        timestamp = time.time()
        
        log_entry = {
            'timestamp': timestamp,
            'session_id': self.current_session_id,
            'optimization_type': 'training',
            'step': step,
            'fitness_score': fitness_score,
            'best_params': best_params,
            'training_stats': training_stats
        }
        
        self.optimization_logs['training'].append(log_entry)
        self.metric_history['training_fitness'].append(fitness_score)
        
        if self.enable_debug:
            print(f"[OptMon] Training Step {step}: Fitness={fitness_score:.4f}, "
                  f"Best LR={best_params.get('learning_rate', 0):.6f}")
    
    def log_reward_optimization(self,
                              step: int,
                              fitness_score: float,
                              best_params: Dict[str, Any],
                              behavioral_stats: Dict[str, float]):
        """Log reward parameter optimization step"""
        
        timestamp = time.time()
        
        log_entry = {
            'timestamp': timestamp,
            'session_id': self.current_session_id,
            'optimization_type': 'reward',
            'step': step,
            'fitness_score': fitness_score,
            'best_params': best_params,
            'behavioral_stats': behavioral_stats
        }
        
        self.optimization_logs['reward'].append(log_entry)
        self.metric_history['reward_fitness'].append(fitness_score)
        
        if self.enable_debug:
            print(f"[OptMon] Reward Step {step}: Fitness={fitness_score:.4f}, "
                  f"Best Exploration Bonus={best_params.get('exploration_bonus', 0):.2f}")
    
    def log_multi_metric_evaluation(self,
                                  step: int,
                                  overall_fitness: float,
                                  objective_scores: Dict[str, float],
                                  detailed_metrics: Dict[str, float]):
        """Log multi-metric fitness evaluation"""
        
        timestamp = time.time()
        
        log_entry = {
            'timestamp': timestamp,
            'session_id': self.current_session_id,
            'evaluation_type': 'multi_metric',
            'step': step,
            'overall_fitness': overall_fitness,
            'objective_scores': objective_scores,
            'detailed_metrics': detailed_metrics
        }
        
        self.performance_logs['multi_metric'].append(log_entry)
        self.metric_history['overall_fitness'].append(overall_fitness)
        
        # Log individual objective scores
        for objective, score in objective_scores.items():
            if f'{objective}_score' in self.metric_history:
                self.metric_history[f'{objective}_score'].append(score)
        
        if self.enable_debug:
            print(f"[OptMon] MultiMetric Step {step}: Overall={overall_fitness:.4f}, "
                  f"Perf={objective_scores.get('performance', 0):.3f}, "
                  f"Safety={objective_scores.get('safety', 0):.3f}")
    
    def log_parameter_update(self,
                           component: str,
                           old_params: Dict[str, Any],
                           new_params: Dict[str, Any],
                           improvement: float):
        """Log parameter update event"""
        
        timestamp = time.time()
        
        log_entry = {
            'timestamp': timestamp,
            'session_id': self.current_session_id,
            'event_type': 'parameter_update',
            'component': component,
            'old_params': old_params,
            'new_params': new_params,
            'improvement': improvement
        }
        
        self.parameter_logs[component].append(log_entry)
        
        if self.enable_debug:
            print(f"[OptMon] {component.upper()} params updated, improvement: {improvement:+.4f}")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get comprehensive optimization summary"""
        
        current_time = time.time()
        session_duration = current_time - self.session_start_time
        
        summary = {
            'session_id': self.current_session_id,
            'session_duration_hours': session_duration / 3600,
            'optimization_components': {
                'es': {
                    'evaluations': len(self.optimization_logs['es']),
                    'best_fitness': max([log['fitness_score'] for log in self.optimization_logs['es']], default=0),
                    'recent_fitness': list(self.metric_history['es_fitness'])[-1] if self.metric_history['es_fitness'] else 0
                },
                'training': {
                    'evaluations': len(self.optimization_logs['training']),
                    'best_fitness': max([log['fitness_score'] for log in self.optimization_logs['training']], default=0),
                    'recent_fitness': list(self.metric_history['training_fitness'])[-1] if self.metric_history['training_fitness'] else 0
                },
                'reward': {
                    'evaluations': len(self.optimization_logs['reward']),
                    'best_fitness': max([log['fitness_score'] for log in self.optimization_logs['reward']], default=0),
                    'recent_fitness': list(self.metric_history['reward_fitness'])[-1] if self.metric_history['reward_fitness'] else 0
                }
            },
            'multi_metric': {
                'evaluations': len(self.performance_logs['multi_metric']),
                'best_overall_fitness': max([log['overall_fitness'] for log in self.performance_logs['multi_metric']], default=0),
                'recent_overall_fitness': list(self.metric_history['overall_fitness'])[-1] if self.metric_history['overall_fitness'] else 0,
                'objective_trends': self._calculate_objective_trends()
            }
        }
        
        return summary
    
    def _calculate_objective_trends(self) -> Dict[str, float]:
        """Calculate recent trends in objective scores"""
        
        trends = {}
        
        for objective in ['performance', 'safety', 'efficiency', 'robustness']:
            key = f'{objective}_score'
            if key in self.metric_history and len(self.metric_history[key]) >= 10:
                recent_scores = list(self.metric_history[key])[-10:]
                trend = (recent_scores[-1] - recent_scores[0]) / max(len(recent_scores) - 1, 1)
                trends[objective] = trend
            else:
                trends[objective] = 0.0
        
        return trends
    
    def export_optimization_report(self, report_path: Optional[str] = None) -> str:
        """Export comprehensive optimization report"""
        
        if report_path is None:
            report_path = os.path.join(self.log_dir, f"optimization_report_{self.current_session_id}.json")
        
        summary = self.get_optimization_summary()
        
        # Add detailed statistics
        report_data = {
            **summary,
            'detailed_logs': {
                'es_optimization': self.optimization_logs['es'][-50:],  # Last 50 entries
                'training_optimization': self.optimization_logs['training'][-50:],
                'reward_optimization': self.optimization_logs['reward'][-50:],
                'multi_metric_evaluation': self.performance_logs['multi_metric'][-100:]
            },
            'parameter_updates': {
                component: logs[-20:] for component, logs in self.parameter_logs.items()
            },
            'metric_statistics': self._calculate_metric_statistics()
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        if self.enable_debug:
            print(f"[OptMon] Exported optimization report to: {report_path}")
        
        return report_path
    
    def _calculate_metric_statistics(self) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for all tracked metrics"""
        
        statistics = {}
        
        for metric_name, history in self.metric_history.items():
            if len(history) > 0:
                history_list = list(history)
                statistics[metric_name] = {
                    'count': len(history_list),
                    'mean': np.mean(history_list),
                    'std': np.std(history_list),
                    'min': np.min(history_list),
                    'max': np.max(history_list),
                    'recent_mean': np.mean(history_list[-20:]) if len(history_list) >= 20 else np.mean(history_list),
                    'trend': self._calculate_trend(history_list)
                }
            else:
                statistics[metric_name] = {
                    'count': 0,
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'recent_mean': 0.0,
                    'trend': 0.0
                }
        
        return statistics
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend (slope) of values over time"""
        
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        trend = np.polyfit(x, values, 1)[0]  # Linear slope
        
        return float(trend)
    
    def _write_logs_to_file(self):
        """Write recent logs to file"""
        
        try:
            # Write session log entries
            with open(self.session_log_file, 'a') as f:
                # Write any unwritten optimization logs
                for log_type, logs in self.optimization_logs.items():
                    for log_entry in logs:
                        if log_entry.get('written', False):
                            continue
                        f.write(json.dumps(log_entry, default=str) + '\n')
                        log_entry['written'] = True
                
                # Write any unwritten performance logs
                for log_type, logs in self.performance_logs.items():
                    for log_entry in logs:
                        if log_entry.get('written', False):
                            continue
                        f.write(json.dumps(log_entry, default=str) + '\n')
                        log_entry['written'] = True
            
            self.last_log_update = time.time()
            
            if self.enable_debug:
                print(f"[OptMon] Updated log files at {datetime.now().strftime('%H:%M:%S')}")
                
        except Exception as e:
            if self.enable_debug:
                print(f"[OptMon] Failed to write logs: {e}")
    
    def setup_visualization(self):
        """Setup real-time visualization if matplotlib available"""
        
        if not MATPLOTLIB_AVAILABLE:
            return
        
        try:
            # Create figure with subplots for different metrics
            self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
            self.fig.suptitle(f'Optimization Monitoring - Session {self.current_session_id}')
            
            # Configure subplots
            self.axes[0, 0].set_title('ES Optimization Fitness')
            self.axes[0, 0].set_xlabel('Generation')
            self.axes[0, 0].set_ylabel('Fitness Score')
            
            self.axes[0, 1].set_title('Training Optimization Fitness')
            self.axes[0, 1].set_xlabel('Step')
            self.axes[0, 1].set_ylabel('Fitness Score')
            
            self.axes[1, 0].set_title('Reward Optimization Fitness')
            self.axes[1, 0].set_xlabel('Step')
            self.axes[1, 0].set_ylabel('Fitness Score')
            
            self.axes[1, 1].set_title('Multi-Objective Scores')
            self.axes[1, 1].set_xlabel('Evaluation')
            self.axes[1, 1].set_ylabel('Score')
            
            # Initialize empty line plots
            self.plot_lines = {}
            
            plt.tight_layout()
            
            if self.enable_debug:
                print("[OptMon] Visualization setup complete")
                
        except Exception as e:
            if self.enable_debug:
                print(f"[OptMon] Visualization setup failed: {e}")
    
    def update_visualization(self):
        """Update real-time visualization plots"""
        
        if not MATPLOTLIB_AVAILABLE or not hasattr(self, 'fig'):
            return
        
        try:
            # Update ES optimization plot
            if self.metric_history['es_fitness']:
                es_fitness = list(self.metric_history['es_fitness'])
                self.axes[0, 0].clear()
                self.axes[0, 0].plot(es_fitness, 'b-', label='ES Fitness')
                self.axes[0, 0].set_title('ES Optimization Fitness')
                self.axes[0, 0].legend()
            
            # Update training optimization plot
            if self.metric_history['training_fitness']:
                training_fitness = list(self.metric_history['training_fitness'])
                self.axes[0, 1].clear()
                self.axes[0, 1].plot(training_fitness, 'g-', label='Training Fitness')
                self.axes[0, 1].set_title('Training Optimization Fitness')
                self.axes[0, 1].legend()
            
            # Update reward optimization plot
            if self.metric_history['reward_fitness']:
                reward_fitness = list(self.metric_history['reward_fitness'])
                self.axes[1, 0].clear()
                self.axes[1, 0].plot(reward_fitness, 'r-', label='Reward Fitness')
                self.axes[1, 0].set_title('Reward Optimization Fitness')
                self.axes[1, 0].legend()
            
            # Update multi-objective plot
            if self.metric_history['overall_fitness']:
                overall_fitness = list(self.metric_history['overall_fitness'])
                performance_scores = list(self.metric_history['performance_score'])
                safety_scores = list(self.metric_history['safety_score'])
                
                self.axes[1, 1].clear()
                self.axes[1, 1].plot(overall_fitness, 'k-', label='Overall', linewidth=2)
                if performance_scores:
                    self.axes[1, 1].plot(performance_scores, 'b--', label='Performance')
                if safety_scores:
                    self.axes[1, 1].plot(safety_scores, 'r--', label='Safety')
                
                self.axes[1, 1].set_title('Multi-Objective Scores')
                self.axes[1, 1].legend()
            
            plt.tight_layout()
            plt.pause(0.01)  # Non-blocking update
            
        except Exception as e:
            if self.enable_debug:
                print(f"[OptMon] Visualization update failed: {e}")
    
    def save_plots(self):
        """Save current plots to files"""
        
        if not MATPLOTLIB_AVAILABLE or not hasattr(self, 'fig'):
            return
        
        try:
            plot_path = os.path.join(self.log_dir, "plots", f"optimization_plots_{self.current_session_id}.png")
            self.fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            
            if self.enable_debug:
                print(f"[OptMon] Plots saved to: {plot_path}")
                
        except Exception as e:
            if self.enable_debug:
                print(f"[OptMon] Failed to save plots: {e}")
    
    def cleanup(self):
        """Clean up monitoring session"""
        
        # Final log write
        self._write_logs_to_file()
        
        # Export final report
        final_report = self.export_optimization_report()
        
        # Save plots if available
        self.save_plots()
        
        # Close matplotlib figures
        if MATPLOTLIB_AVAILABLE and hasattr(self, 'fig'):
            plt.close(self.fig)
        
        if self.enable_debug:
            print(f"[OptMon] Session {self.current_session_id} cleanup complete")
            print(f"[OptMon] Final report: {final_report}")
        
        return final_report