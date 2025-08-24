#!/usr/bin/env python3
"""
Model Transfer Utilities for Rover Deployment
Transfer models between ultra-fast training system and rover ES hybrid system
"""

import torch
import numpy as np
import os
import shutil
import time
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

from rover_compatible_model import RoverDepthExplorationNet, create_rover_compatible_model
from model_utils import load_trained_model


class RoverModelTransfer:
    """Handles model transfers between systems"""
    
    def __init__(self, rover_model_dir: str = "../../../models/simulation", 
                 ultra_fast_model_dir: str = "./models/ultra_fast"):
        self.rover_model_dir = Path(rover_model_dir)
        self.ultra_fast_model_dir = Path(ultra_fast_model_dir)
        self.logger = logging.getLogger(__name__)
        
        # Ensure directories exist
        self.rover_model_dir.mkdir(parents=True, exist_ok=True)
        
    def convert_ultra_fast_to_rover(self, ultra_fast_model_path: str, 
                                   rover_output_name: str = None) -> str:
        """
        Convert ultra-fast model to rover ES format
        
        Args:
            ultra_fast_model_path: Path to ultra-fast model checkpoint
            rover_output_name: Name for rover model (auto-generated if None)
            
        Returns:
            Path to converted rover model
        """
        self.logger.info(f"Converting ultra-fast model: {ultra_fast_model_path}")
        
        # Load ultra-fast checkpoint
        if not os.path.exists(ultra_fast_model_path):
            raise FileNotFoundError(f"Ultra-fast model not found: {ultra_fast_model_path}")
            
        checkpoint = torch.load(ultra_fast_model_path, map_location='cpu')
        
        # Create rover-compatible model
        rover_model = RoverDepthExplorationNet(
            stacked_frames=1,
            extra_proprio=13,
            use_quantization=False,  # Disable for transfer
            use_sparsity=False
        )
        
        # Try to load weights
        try:
            # Check if architectures are compatible
            ultra_fast_state = checkpoint['model_state_dict']
            rover_state = rover_model.state_dict()
            
            # Compare layer shapes
            compatible = True
            for name, param in rover_state.items():
                if name in ultra_fast_state:
                    if param.shape != ultra_fast_state[name].shape:
                        self.logger.warning(f"Shape mismatch for {name}: "
                                          f"rover {param.shape} vs ultra-fast {ultra_fast_state[name].shape}")
                        compatible = False
                else:
                    self.logger.warning(f"Layer {name} missing in ultra-fast model")
                    compatible = False
            
            if compatible:
                rover_model.load_state_dict(ultra_fast_state)
                self.logger.info("✓ Direct weight loading successful")
            else:
                self.logger.warning("Architectures incompatible - using transfer learning")
                self._transfer_compatible_weights(ultra_fast_state, rover_model)
                
        except Exception as e:
            self.logger.error(f"Weight loading failed: {e}")
            self.logger.info("Using randomly initialized rover model")
        
        # Generate output filename
        if rover_output_name is None:
            timestamp = int(time.time())
            rover_output_name = f"exploration_model_depth_es_{timestamp}.pth"
        elif not rover_output_name.endswith('.pth'):
            rover_output_name += '.pth'
            
        # Create rover-format checkpoint
        rover_checkpoint = {
            'model_state_dict': rover_model.state_dict(),
            'generation': checkpoint.get('generation', 0),
            'best_fitness': checkpoint.get('fitness', checkpoint.get('best_fitness', -999.0)),
            'fitness_history': checkpoint.get('training_history', []),
            'elite_individuals': [],  # Will be rebuilt during ES training
            'elite_fitness_scores': [],
            'sigma': 0.1,  # Default ES sigma
            'stagnation_counter': 0,
            'momentum': None,
            'velocity': None,
            'update_step': 0,
            'diversity_history': [],
            # Transfer metadata
            'transfer_info': {
                'source': 'ultra_fast_training',
                'original_model': ultra_fast_model_path,
                'transfer_timestamp': time.time(),
                'optimization_type': checkpoint.get('optimization_type', 'Unknown')
            }
        }
        
        # Save rover model
        rover_model_path = self.rover_model_dir / rover_output_name
        torch.save(rover_checkpoint, rover_model_path)
        
        # Update latest symlink
        latest_path = self.rover_model_dir / "exploration_model_depth_es_latest.pth"
        if latest_path.exists():
            latest_path.unlink()
        latest_path.symlink_to(rover_output_name)
        
        self.logger.info(f"✓ Rover model saved: {rover_model_path}")
        self.logger.info(f"✓ Latest symlink updated: {latest_path}")
        
        return str(rover_model_path)
        
    def _transfer_compatible_weights(self, source_state: Dict, target_model: torch.nn.Module):
        """Transfer compatible weights between different architectures"""
        target_state = target_model.state_dict()
        
        transferred = 0
        for name, param in target_state.items():
            if name in source_state and param.shape == source_state[name].shape:
                param.copy_(source_state[name])
                transferred += 1
                self.logger.debug(f"Transferred: {name}")
            else:
                self.logger.debug(f"Skipped: {name} (incompatible or missing)")
                
        self.logger.info(f"Transferred {transferred}/{len(target_state)} layers")
        
    def create_rover_training_config(self, model_path: str, 
                                    ultra_fast_results: Optional[Dict] = None) -> Dict:
        """
        Create rover training configuration based on ultra-fast results
        
        Args:
            model_path: Path to rover model
            ultra_fast_results: Results from ultra-fast training
            
        Returns:
            Configuration dict for rover training
        """
        config = {
            # ES hyperparameters optimized based on ultra-fast results
            "population_size": 12,  # Smaller for real hardware
            "sigma": 0.05,  # Smaller for pre-trained model
            "learning_rate": 0.005,  # Conservative for real robot
            "enable_bayesian_optimization": True,
            
            # Training parameters
            "max_exploration_time": 300,  # 5 minutes per episode
            "safety_distance": 0.3,       # Conservative safety
            "max_speed": 0.15,           # Conservative speed
            
            # Model info
            "model_path": model_path,
            "model_type": "depth_es",
            "stacked_frames": 1,
            "extra_proprio": 13,
            
            # Hardware optimization
            "enable_rknn_conversion": True,
            "rknn_conversion_frequency": 10,  # Convert every 10 generations
            "enable_quantization": True,
            
            # Transfer learning settings
            "initial_sigma_decay": 0.95,  # Reduce exploration initially  
            "elite_preservation": True,
            "transfer_learning_mode": True,
        }
        
        # Adjust based on ultra-fast results if available
        if ultra_fast_results:
            best_fitness = ultra_fast_results.get('best_fitness', -999)
            if best_fitness > -10:  # Model is somewhat trained
                config["sigma"] = 0.03  # Even smaller sigma
                config["learning_rate"] = 0.003
                config["population_size"] = 8  # Smaller population
            
            # Copy training history for warm start
            config["fitness_history"] = ultra_fast_results.get('convergence_history', [])
            
        return config
        
    def prepare_rover_deployment(self, ultra_fast_model_path: str, 
                                deployment_name: str = None) -> Dict:
        """
        Prepare complete rover deployment package
        
        Args:
            ultra_fast_model_path: Path to trained ultra-fast model
            deployment_name: Name for deployment (auto-generated if None)
            
        Returns:
            Deployment info dict
        """
        if deployment_name is None:
            deployment_name = f"rover_deployment_{int(time.time())}"
            
        self.logger.info(f"Preparing rover deployment: {deployment_name}")
        
        # Create deployment directory
        deploy_dir = self.rover_model_dir / deployment_name
        deploy_dir.mkdir(exist_ok=True)
        
        # Convert model
        rover_model_path = self.convert_ultra_fast_to_rover(
            ultra_fast_model_path, 
            f"{deployment_name}_model.pth"
        )
        
        # Copy model to deployment directory
        deployed_model = deploy_dir / "exploration_model_depth_es_latest.pth"
        shutil.copy2(rover_model_path, deployed_model)
        
        # Load ultra-fast results
        ultra_fast_results = {}
        try:
            results_path = Path(ultra_fast_model_path).parent / "bayesian_results.json"
            if results_path.exists():
                with open(results_path, 'r') as f:
                    ultra_fast_results = json.load(f)
        except Exception as e:
            self.logger.warning(f"Could not load ultra-fast results: {e}")
        
        # Create training config
        config = self.create_rover_training_config(str(deployed_model), ultra_fast_results)
        config_path = deploy_dir / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Create deployment script
        self._create_deployment_script(deploy_dir, config)
        
        # Create deployment info
        deployment_info = {
            "deployment_name": deployment_name,
            "deployment_dir": str(deploy_dir),
            "model_path": str(deployed_model),
            "config_path": str(config_path),
            "source_model": ultra_fast_model_path,
            "ultra_fast_results": ultra_fast_results,
            "deployment_timestamp": time.time(),
            "ready_for_rover": True
        }
        
        # Save deployment info
        info_path = deploy_dir / "deployment_info.json"
        with open(info_path, 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        self.logger.info(f"✓ Rover deployment ready: {deploy_dir}")
        self.logger.info("To deploy on rover:")
        self.logger.info(f"  1. Copy {deploy_dir} to rover")
        self.logger.info(f"  2. Run: ./deploy_to_rover.sh")
        
        return deployment_info
        
    def _create_deployment_script(self, deploy_dir: Path, config: Dict):
        """Create deployment script for rover"""
        script_content = f"""#!/bin/bash
# Rover Model Deployment Script
# Generated automatically from ultra-fast training

echo "🤖 Deploying ultra-fast trained model to rover..."

# Configuration from ultra-fast training results
POPULATION_SIZE={config['population_size']}
SIGMA={config['sigma']}
LEARNING_RATE={config['learning_rate']}
MAX_SPEED={config['max_speed']}
SAFETY_DISTANCE={config['safety_distance']}

# Copy model to rover models directory
cp exploration_model_depth_es_latest.pth ../../../models/simulation/

# Launch rover ES hybrid training with pre-trained model
echo "Launching ES hybrid training with ultra-fast pre-trained model..."

cd ../../../

# Run rover training with optimized parameters
./start_npu_exploration_depth.sh es_hybrid \\
    $MAX_SPEED \\
    300 \\
    $SAFETY_DISTANCE

echo "✓ Rover deployment complete! Model is now evolving on real hardware."
"""
        
        script_path = deploy_dir / "deploy_to_rover.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        script_path.chmod(0o755)  # Make executable
        
    def verify_rover_compatibility(self, model_path: str) -> bool:
        """
        Verify that a model is compatible with rover system
        
        Args:
            model_path: Path to model to verify
            
        Returns:
            True if compatible, False otherwise
        """
        try:
            self.logger.info(f"Verifying rover compatibility: {model_path}")
            
            # Load model
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Check required keys
            required_keys = ['model_state_dict', 'generation', 'best_fitness']
            for key in required_keys:
                if key not in checkpoint:
                    self.logger.error(f"Missing required key: {key}")
                    return False
            
            # Try to create model with saved state
            rover_model = RoverDepthExplorationNet(
                stacked_frames=1,
                extra_proprio=13,
                use_quantization=False,
                use_sparsity=False
            )
            rover_model.load_state_dict(checkpoint['model_state_dict'])
            
            # Test forward pass with rover dimensions
            depth_test = torch.randn(1, 1, 160, 288)  # [batch, channels, height, width]
            proprio_test = torch.randn(1, 16)  # [batch, features]
            
            with torch.no_grad():
                output = rover_model(depth_test, proprio_test)
                
            if output.shape != (1, 3):
                self.logger.error(f"Wrong output shape: {output.shape}, expected (1, 3)")
                return False
                
            self.logger.info("✓ Model is rover-compatible")
            return True
            
        except Exception as e:
            self.logger.error(f"Compatibility check failed: {e}")
            return False
            
    def list_available_models(self) -> Dict:
        """List available models for transfer"""
        models = {
            "ultra_fast": [],
            "rover": []
        }
        
        # Check ultra-fast models
        if self.ultra_fast_model_dir.exists():
            for model_file in self.ultra_fast_model_dir.glob("*.pth"):
                try:
                    checkpoint = torch.load(model_file, map_location='cpu')
                    models["ultra_fast"].append({
                        "path": str(model_file),
                        "name": model_file.name,
                        "fitness": checkpoint.get('fitness', checkpoint.get('best_fitness', 'Unknown')),
                        "timestamp": checkpoint.get('save_timestamp', 'Unknown')
                    })
                except:
                    pass
        
        # Check rover models  
        if self.rover_model_dir.exists():
            for model_file in self.rover_model_dir.glob("exploration_model_depth_es_*.pth"):
                try:
                    checkpoint = torch.load(model_file, map_location='cpu')
                    models["rover"].append({
                        "path": str(model_file),
                        "name": model_file.name,
                        "generation": checkpoint.get('generation', 'Unknown'),
                        "fitness": checkpoint.get('best_fitness', 'Unknown')
                    })
                except:
                    pass
        
        return models


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Rover model transfer utilities")
    parser.add_argument("--convert", help="Convert ultra-fast model to rover format")
    parser.add_argument("--output", help="Output name for converted model")
    parser.add_argument("--deploy", help="Prepare complete rover deployment")
    parser.add_argument("--verify", help="Verify rover compatibility of model")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--rover-models", default="../../../models/simulation", 
                       help="Rover models directory")
    parser.add_argument("--ultra-fast-models", default="./models/ultra_fast",
                       help="Ultra-fast models directory")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create transfer utility
    transfer = RoverModelTransfer(args.rover_models, args.ultra_fast_models)
    
    if args.list:
        models = transfer.list_available_models()
        print("Available Models:")
        print("\nUltra-Fast Models:")
        for model in models["ultra_fast"]:
            print(f"  {model['name']} - Fitness: {model['fitness']}")
        print("\nRover Models:")
        for model in models["rover"]:
            print(f"  {model['name']} - Gen: {model['generation']}, Fitness: {model['fitness']}")
            
    elif args.convert:
        result = transfer.convert_ultra_fast_to_rover(args.convert, args.output)
        print(f"✓ Converted model saved: {result}")
        
    elif args.deploy:
        deployment_info = transfer.prepare_rover_deployment(args.deploy)
        print(f"✓ Deployment ready: {deployment_info['deployment_dir']}")
        
    elif args.verify:
        is_compatible = transfer.verify_rover_compatibility(args.verify)
        print(f"Model compatibility: {'✓ PASSED' if is_compatible else '✗ FAILED'}")
        
    else:
        print("Use --list to see available models")
        print("Use --convert <model> to convert ultra-fast model to rover format")
        print("Use --deploy <model> to prepare complete rover deployment")


if __name__ == "__main__":
    main()