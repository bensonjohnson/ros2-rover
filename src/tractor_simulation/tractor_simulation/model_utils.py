#!/usr/bin/env python3
"""
Model utilities for loading and managing trained models
"""

import torch
from pathlib import Path
from typing import Optional, Dict, Tuple
import json
import logging

from optimized_model import OptimizedDepthModel, UltraFastTrainer


def load_trained_model(model_path: str, device: str = "cuda") -> Tuple[OptimizedDepthModel, Dict]:
    """
    Load a trained model from checkpoint
    
    Args:
        model_path: Path to the .pth model file
        device: Device to load model on
        
    Returns:
        model: Loaded model
        metadata: Model metadata and training info
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    logger = logging.getLogger(__name__)
    logger.info(f"Loading model from {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract architecture info
    architecture = checkpoint.get('architecture', {
        'depth_dim': 64,
        'proprio_dim': 8,
        'hidden_dim': 128,
        'action_dim': 2
    })
    
    # Create model with same architecture
    model = OptimizedDepthModel(
        depth_dim=architecture['depth_dim'],
        proprio_dim=architecture['proprio_dim'],
        hidden_dim=architecture['hidden_dim'],
        action_dim=architecture['action_dim'],
        use_quantization=True,
        use_sparsity=True
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Extract metadata
    metadata = {
        'fitness': checkpoint.get('fitness', None),
        'optimization_type': checkpoint.get('optimization_type', 'Unknown'),
        'training_history': checkpoint.get('training_history', []),
        'save_timestamp': checkpoint.get('save_timestamp', None),
        'architecture': architecture
    }
    
    logger.info(f"Model loaded successfully. Best fitness: {metadata['fitness']:.4f}")
    
    return model, metadata


def create_model_inference_wrapper(model_path: str, device: str = "cuda"):
    """
    Create a simple inference wrapper for a trained model
    
    Args:
        model_path: Path to the .pth model file
        device: Device to run inference on
        
    Returns:
        Inference function that takes (depth, proprio) and returns actions
    """
    model, metadata = load_trained_model(model_path, device)
    
    def inference_fn(depth: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """
        Run inference on depth and proprioceptive data
        
        Args:
            depth: [batch_size, 64] or [64] - depth sensor data
            proprio: [batch_size, 8] or [8] - proprioceptive data
            
        Returns:
            actions: [batch_size, 2] or [2] - predicted actions
        """
        # Handle single samples
        single_sample = False
        if depth.dim() == 1:
            depth = depth.unsqueeze(0)
            single_sample = True
        if proprio.dim() == 1:
            proprio = proprio.unsqueeze(0)
            single_sample = True
            
        # Move to device
        depth = depth.to(device)
        proprio = proprio.to(device)
        
        # Inference
        with torch.no_grad():
            actions = model.get_actions(depth, proprio)
            
        # Handle single sample output
        if single_sample:
            actions = actions.squeeze(0)
            
        return actions
        
    return inference_fn, metadata


def compare_models(model_paths: list, test_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
    """
    Compare multiple trained models
    
    Args:
        model_paths: List of paths to model files
        test_data: Optional (depth, proprio) test data
        
    Returns:
        Comparison results
    """
    results = []
    
    for path in model_paths:
        try:
            model, metadata = load_trained_model(path)
            
            result = {
                'path': str(path),
                'fitness': metadata['fitness'],
                'optimization_type': metadata['optimization_type'],
                'timestamp': metadata['save_timestamp']
            }
            
            # Run test if provided
            if test_data is not None:
                depth_test, proprio_test = test_data
                inference_fn, _ = create_model_inference_wrapper(path)
                
                with torch.no_grad():
                    actions = inference_fn(depth_test, proprio_test)
                    result['test_action_mean'] = actions.mean().item()
                    result['test_action_std'] = actions.std().item()
                    
            results.append(result)
            
        except Exception as e:
            results.append({
                'path': str(path),
                'error': str(e)
            })
            
    return results


def export_model_to_onnx(model_path: str, output_path: str, device: str = "cuda"):
    """
    Export a trained model to ONNX format for deployment
    
    Args:
        model_path: Path to the .pth model file
        output_path: Path to save .onnx file
        device: Device for export
    """
    model, metadata = load_trained_model(model_path, device)
    
    # Create dummy inputs
    dummy_depth = torch.randn(1, 64, device=device)
    dummy_proprio = torch.randn(1, 8, device=device)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_depth, dummy_proprio),
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['depth', 'proprio'],
        output_names=['actions', 'values'],
        dynamic_axes={
            'depth': {0: 'batch_size'},
            'proprio': {0: 'batch_size'},
            'actions': {0: 'batch_size'},
            'values': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to ONNX: {output_path}")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Model utilities")
    parser.add_argument("--load", help="Path to model to load")
    parser.add_argument("--compare", nargs="+", help="Paths to models to compare")
    parser.add_argument("--export-onnx", help="Export model to ONNX")
    parser.add_argument("--device", default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    if args.load:
        model, metadata = load_trained_model(args.load, args.device)
        print(f"Model loaded: {metadata}")
        
    if args.compare:
        results = compare_models(args.compare)
        print("Model comparison:")
        for result in results:
            print(f"  {result}")
            
    if args.export_onnx and args.load:
        export_model_to_onnx(args.load, args.export_onnx, args.device)