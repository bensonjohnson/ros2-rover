#!/usr/bin/env python3
"""
Training Monitor for Overtraining Detection
Provides real-time monitoring and alerts for potential overtraining
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from typing import Dict, List, Optional
import json
import time
from datetime import datetime

class TrainingMonitor:
    """Monitor training progress and detect potential overtraining"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics_history = {
            'rewards': deque(maxlen=window_size),
            'behavioral_diversity': deque(maxlen=window_size),
            'overtraining_risk': deque(maxlen=window_size),
            'exploration_efficiency': deque(maxlen=window_size),
            'timestamps': deque(maxlen=window_size)
        }
        
        self.alerts = []
        self.training_start_time = time.time()
        self.last_checkpoint_time = time.time()
        
    def update(self, reward_calculator, episode_reward: float, step: int):
        """Update monitoring metrics"""
        current_time = time.time()
        
        # Get current statistics
        stats = reward_calculator.get_reward_statistics()
        
        # Update history
        self.metrics_history['rewards'].append(episode_reward)
        self.metrics_history['behavioral_diversity'].append(stats.get('behavioral_diversity', 0.0))
        self.metrics_history['overtraining_risk'].append(stats.get('potential_overtraining_score', 0.0))
        self.metrics_history['exploration_efficiency'].append(stats.get('unique_areas', 0) / max(stats.get('total_steps', 1), 1))
        self.metrics_history['timestamps'].append(current_time)
        
        # Check for alerts
        self._check_alerts(stats, step)
        
        return self._generate_report(stats, step)
    
    def _check_alerts(self, stats: Dict, step: int):
        """Check for potential overtraining conditions"""
        current_time = time.time()
        
        # Alert 1: Low behavioral diversity
        diversity = stats.get('behavioral_diversity', 1.0)
        if diversity < 0.2 and len(self.metrics_history['rewards']) > 100:
            self.alerts.append({
                'type': 'LOW_DIVERSITY',
                'step': step,
                'message': f'Low behavioral diversity: {diversity:.3f}',
                'severity': 'WARNING',
                'timestamp': current_time
            })
        
        # Alert 2: High overtraining risk
        risk = stats.get('potential_overtraining_score', 0.0)
        if risk > 0.7:
            self.alerts.append({
                'type': 'HIGH_OVERTRAINING_RISK',
                'step': step,
                'message': f'High overtraining risk: {risk:.3f}',
                'severity': 'CRITICAL',
                'timestamp': current_time
            })
        
        # Alert 3: Reward plateau with poor diversity
        if len(self.metrics_history['rewards']) > 200:
            rewards_list = list(self.metrics_history['rewards'])
            recent_rewards = rewards_list[-100:]
            earlier_rewards = rewards_list[-200:-100]
            
            improvement = np.mean(recent_rewards) - np.mean(earlier_rewards)
            if improvement < 0.5 and diversity < 0.3:
                self.alerts.append({
                    'type': 'REWARD_PLATEAU',
                    'step': step,
                    'message': f'Reward plateau with low diversity: improvement={improvement:.3f}, diversity={diversity:.3f}',
                    'severity': 'WARNING',
                    'timestamp': current_time
                })
        
        # Alert 4: Excessive area revisiting
        max_visits = max(stats.get('area_visit_count', {}).values()) if stats.get('area_visit_count') else 0
        if max_visits > 15:
            self.alerts.append({
                'type': 'EXCESSIVE_REVISITING',
                'step': step,
                'message': f'Excessive area revisiting: max_visits={max_visits}',
                'severity': 'WARNING',
                'timestamp': current_time
            })
    
    def _generate_report(self, stats: Dict, step: int) -> Dict:
        """Generate a comprehensive training report"""
        current_time = time.time()
        training_duration = current_time - self.training_start_time
        
        report = {
            'step': step,
            'training_duration_hours': training_duration / 3600,
            'current_stats': stats,
            'recent_alerts': [alert for alert in self.alerts if current_time - alert['timestamp'] < 3600],  # Last hour
            'health_indicators': self._get_health_indicators(stats),
            'recommendations': self._get_recommendations(stats)
        }
        
        return report
    
    def _get_health_indicators(self, stats: Dict) -> Dict:
        """Get training health indicators"""
        diversity = stats.get('behavioral_diversity', 0.0)
        risk = stats.get('potential_overtraining_score', 0.0)
        exploration_rate = stats.get('unique_areas', 0) / max(stats.get('total_steps', 1), 1)
        
        return {
            'diversity_health': 'GOOD' if diversity > 0.4 else 'POOR' if diversity < 0.2 else 'FAIR',
            'overtraining_health': 'GOOD' if risk < 0.3 else 'CRITICAL' if risk > 0.7 else 'WARNING',
            'exploration_health': 'GOOD' if exploration_rate > 0.01 else 'POOR' if exploration_rate < 0.005 else 'FAIR',
            'overall_health': self._calculate_overall_health(diversity, risk, exploration_rate)
        }
    
    def _calculate_overall_health(self, diversity: float, risk: float, exploration_rate: float) -> str:
        """Calculate overall training health"""
        score = 0
        
        if diversity > 0.4:
            score += 1
        if risk < 0.3:
            score += 1
        if exploration_rate > 0.01:
            score += 1
        
        if score >= 2:
            return 'GOOD'
        elif score == 1:
            return 'FAIR'
        else:
            return 'POOR'
    
    def _get_recommendations(self, stats: Dict) -> List[str]:
        """Get training recommendations based on current state"""
        recommendations = []
        
        diversity = stats.get('behavioral_diversity', 0.0)
        risk = stats.get('potential_overtraining_score', 0.0)
        
        if diversity < 0.2:
            recommendations.append("Increase reward noise or reduce reward magnitude to encourage exploration")
        
        if risk > 0.7:
            recommendations.append("Consider early stopping or curriculum reset")
        
        if len(self.metrics_history['rewards']) > 500:
            rewards_list = list(self.metrics_history['rewards'])
            recent_mean = np.mean(rewards_list[-100:])
            earlier_mean = np.mean(rewards_list[-200:-100])
            
            if recent_mean - earlier_mean < 1.0:
                recommendations.append("Performance plateau detected - consider curriculum progression")
        
        max_visits = max(stats.get('area_visit_count', {}).values()) if stats.get('area_visit_count') else 0
        if max_visits > 10:
            recommendations.append("High area revisiting - increase exploration bonuses or reset tracking")
        
        if not recommendations:
            recommendations.append("Training appears healthy - continue current approach")
        
        return recommendations
    
    def plot_training_progress(self, save_path: Optional[str] = None):
        """Plot training progress metrics"""
        if len(self.metrics_history['rewards']) < 10:
            print("Insufficient data for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress Monitor', fontsize=16)
        
        # Plot 1: Rewards over time
        axes[0, 0].plot(list(self.metrics_history['rewards']))
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
        
        # Plot 2: Behavioral diversity
        axes[0, 1].plot(list(self.metrics_history['behavioral_diversity']))
        axes[0, 1].set_title('Behavioral Diversity')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Diversity Score')
        axes[0, 1].axhline(y=0.3, color='r', linestyle='--', label='Warning Threshold')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot 3: Overtraining risk
        axes[1, 0].plot(list(self.metrics_history['overtraining_risk']))
        axes[1, 0].set_title('Overtraining Risk')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Risk Score')
        axes[1, 0].axhline(y=0.7, color='r', linestyle='--', label='Critical Threshold')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot 4: Exploration efficiency
        axes[1, 1].plot(list(self.metrics_history['exploration_efficiency']))
        axes[1, 1].set_title('Exploration Efficiency')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Unique Areas / Step')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def save_training_log(self, filepath: str):
        """Save detailed training log"""
        log_data = {
            'training_start_time': self.training_start_time,
            'total_episodes': len(self.metrics_history['rewards']),
            'metrics_history': {
                key: list(values) for key, values in self.metrics_history.items()
            },
            'alerts': self.alerts,
            'final_stats': {
                'mean_reward': np.mean(list(self.metrics_history['rewards'])) if self.metrics_history['rewards'] else 0,
                'final_diversity': list(self.metrics_history['behavioral_diversity'])[-1] if self.metrics_history['behavioral_diversity'] else 0,
                'final_risk': list(self.metrics_history['overtraining_risk'])[-1] if self.metrics_history['overtraining_risk'] else 0,
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"Training log saved to {filepath}")
    
    def get_early_stopping_recommendation(self) -> bool:
        """Get recommendation for early stopping"""
        if len(self.metrics_history['rewards']) < 200:
            return False
        
        # Check recent alerts
        current_time = time.time()
        recent_critical_alerts = [
            alert for alert in self.alerts 
            if alert['severity'] == 'CRITICAL' and current_time - alert['timestamp'] < 1800  # Last 30 minutes
        ]
        
        if len(recent_critical_alerts) > 2:
            return True
        
        # Check overtraining risk trend
        risk_list = list(self.metrics_history['overtraining_risk'])
        recent_risk = risk_list[-50:]
        if len(recent_risk) >= 50 and np.mean(recent_risk) > 0.8:
            return True
        
        return False
