#!/usr/bin/env python3
"""
Anti-Overtraining Configuration for Improved Reward System
Provides safe default parameters and monitoring utilities
"""

# Safe default configuration to prevent overtraining
ANTI_OVERTRAINING_CONFIG = {
    # Core reward parameters (simplified and balanced)
    'base_movement_reward': 5.0,
    'forward_progress_bonus': 8.0,
    'exploration_bonus': 10.0,
    'collision_penalty': -20.0,
    'near_collision_penalty': -5.0,
    'unsafe_behavior_penalty': -3.0,
    'smooth_movement_bonus': 1.0,
    'goal_oriented_bonus': 5.0,
    'stagnation_penalty': -2.0,
    
    # Anti-overtraining measures
    'reward_noise_std': 0.1,
    'reward_clip_range': (-30.0, 30.0),
    'reward_smoothing_alpha': 0.1,
    'max_area_revisit_bonus': 3,
    'spinning_threshold': 0.5,
    'behavior_diversity_window': 20,
    
    # Feature toggles
    'enable_reward_smoothing': True,
    'enable_anti_gaming': True,
    'enable_diversity_tracking': True,
}

# Curriculum learning stages
CURRICULUM_STAGES = {
    'beginner': {
        'difficulty_level': 0.3,
        'reward_noise_std': 0.2,
        'exploration_bonus_multiplier': 1.5,
        'description': 'High exploration encouragement, forgiving penalties'
    },
    'intermediate': {
        'difficulty_level': 0.6,
        'reward_noise_std': 0.1,
        'exploration_bonus_multiplier': 1.2,
        'description': 'Balanced exploration and efficiency'
    },
    'advanced': {
        'difficulty_level': 0.9,
        'reward_noise_std': 0.05,
        'exploration_bonus_multiplier': 1.0,
        'description': 'Efficiency focused, minimal noise'
    }
}

# Overtraining detection thresholds
OVERTRAINING_THRESHOLDS = {
    'behavioral_diversity_min': 0.3,
    'reward_improvement_plateau': 1.0,
    'max_area_revisits': 10,
    'training_test_gap_max': 5.0,
    'risk_score_max': 0.7
}

# Early stopping configuration
EARLY_STOPPING_CONFIG = {
    'patience': 100,
    'min_improvement': 1.0,
    'validation_frequency': 50,
    'save_best_model': True
}

def get_safe_config():
    """Get a safe configuration that resists overtraining"""
    return ANTI_OVERTRAINING_CONFIG.copy()

def get_curriculum_config(stage: str):
    """Get configuration for a specific curriculum stage"""
    if stage not in CURRICULUM_STAGES:
        raise ValueError(f"Unknown curriculum stage: {stage}")
    
    base_config = get_safe_config()
    stage_config = CURRICULUM_STAGES[stage]
    
    # Apply stage-specific modifications
    base_config.update(stage_config)
    
    return base_config

def should_progress_curriculum(stats: dict) -> bool:
    """Determine if model is ready to progress to next curriculum stage"""
    if not stats:
        return False
    
    # Check for sufficient behavioral diversity
    diversity = stats.get('behavioral_diversity', 0.0)
    if diversity < 0.4:
        return False
    
    # Check for stable performance
    recent_mean = stats.get('recent_mean', 0.0)
    if recent_mean < 5.0:  # Minimum performance threshold
        return False
    
    # Check for low overtraining risk
    overtraining_risk = stats.get('potential_overtraining_score', 1.0)
    if overtraining_risk > 0.5:
        return False
    
    return True

def get_monitoring_config():
    """Get configuration for monitoring training progress"""
    return {
        'log_frequency': 100,  # Log stats every 100 steps
        'validation_frequency': 500,  # Run validation every 500 steps
        'checkpoint_frequency': 1000,  # Save checkpoint every 1000 steps
        'diversity_check_frequency': 200,  # Check behavioral diversity
        'overtraining_check_frequency': 300,  # Check overtraining risk
    }
