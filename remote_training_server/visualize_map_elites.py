#!/usr/bin/env python3
"""Visualize MAP-Elites archive as a heatmap."""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def load_archive(archive_path: str):
    """Load MAP-Elites archive from JSON file."""
    with open(archive_path, 'r') as f:
        metadata = json.load(f)

    # Load model states from separate directory
    models_dir = Path(archive_path).parent / 'map_elites_models'

    archive = {}
    for model_file in models_dir.glob('cell_*.pt'):
        # Parse cell indices from filename: cell_0_3.pt -> (0, 3)
        parts = model_file.stem.split('_')
        cell_idx = (int(parts[1]), int(parts[2]))

        # For visualization, we just need the cell index
        # Actual model loading would require PyTorch
        archive[cell_idx] = {'exists': True}

    return metadata, archive


def visualize_archive(archive_path: str, output_path: str = None):
    """Create heatmap visualization of MAP-Elites archive."""
    metadata, archive = load_archive(archive_path)

    speed_bins = metadata['speed_bins']
    clearance_bins = metadata['clearance_bins']

    # Create grid for visualization
    n_speed = len(speed_bins) - 1
    n_clearance = len(clearance_bins) - 1

    # Initialize grid (-1 = empty, 0+ = occupied)
    grid = np.full((n_clearance, n_speed), -1.0)

    # Mark occupied cells
    for (speed_idx, clearance_idx) in archive.keys():
        grid[clearance_idx, speed_idx] = 1.0  # Just mark as occupied

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot heatmap
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color='lightgray')

    # Mask empty cells
    masked_grid = np.ma.masked_where(grid < 0, grid)

    im = ax.imshow(masked_grid, cmap=cmap, aspect='auto', origin='lower')

    # Set ticks and labels
    ax.set_xticks(np.arange(n_speed))
    ax.set_yticks(np.arange(n_clearance))

    # Create labels for bins
    speed_labels = [f'{speed_bins[i]:.2f}-{speed_bins[i+1]:.2f}' for i in range(n_speed)]
    clearance_labels = [f'{clearance_bins[i]:.1f}-{clearance_bins[i+1]:.1f}' for i in range(n_clearance)]

    ax.set_xticklabels(speed_labels, rotation=45, ha='right')
    ax.set_yticklabels(clearance_labels)

    ax.set_xlabel('Average Speed (m/s)', fontsize=12)
    ax.set_ylabel('Average Obstacle Clearance (m)', fontsize=12)
    ax.set_title(f'MAP-Elites Archive Coverage\n'
                 f'({len(archive)}/{n_speed*n_clearance} cells filled, '
                 f'{len(archive)/(n_speed*n_clearance)*100:.1f}%)',
                 fontsize=14, fontweight='bold')

    # Add grid
    ax.set_xticks(np.arange(n_speed+1)-0.5, minor=True)
    ax.set_yticks(np.arange(n_clearance+1)-0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)

    # Annotate cells with markers
    for (speed_idx, clearance_idx) in archive.keys():
        ax.text(speed_idx, clearance_idx, '✓',
                ha='center', va='center', color='white', fontsize=16, fontweight='bold')

    # Add legend for behavior profiles
    behaviors = [
        ('Cautious', 0, 3, 'slow + safe'),
        ('Balanced', 2, 2, 'medium + medium'),
        ('Aggressive', 3, 1, 'fast + risky'),
        ('Explorer', 2, 3, 'medium + safe'),
    ]

    for name, speed_idx, clear_idx, desc in behaviors:
        if (speed_idx, clear_idx) in archive:
            ax.add_patch(mpatches.Rectangle((speed_idx-0.45, clear_idx-0.45), 0.9, 0.9,
                                            fill=False, edgecolor='red', linewidth=3))
            ax.text(speed_idx, clear_idx-0.7, name,
                    ha='center', va='top', color='red', fontsize=9, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to {output_path}")
    else:
        plt.show()


def print_archive_stats(archive_path: str):
    """Print detailed archive statistics."""
    with open(archive_path, 'r') as f:
        metadata = json.load(f)

    speed_bins = metadata['speed_bins']
    clearance_bins = metadata['clearance_bins']
    total_cells = (len(speed_bins) - 1) * (len(clearance_bins) - 1)

    print("=" * 60)
    print("MAP-Elites Archive Statistics")
    print("=" * 60)
    print()
    print(f"Archive file: {archive_path}")
    print(f"Total evaluations: {metadata['total_evaluations']}")
    print(f"Archive additions: {metadata['archive_additions']}")
    print(f"Total cells: {total_cells}")
    print()
    print(f"Speed bins: {speed_bins}")
    print(f"Clearance bins: {clearance_bins}")
    print()

    # Load archive to count filled cells
    _, archive = load_archive(archive_path)
    coverage = len(archive) / total_cells

    print(f"Filled cells: {len(archive)}/{total_cells} ({coverage*100:.1f}%)")
    print()

    # List filled cells by behavior
    print("Filled cells:")
    for (speed_idx, clear_idx) in sorted(archive.keys()):
        speed_range = f"{speed_bins[speed_idx]:.2f}-{speed_bins[speed_idx+1]:.2f}"
        clear_range = f"{clearance_bins[clear_idx]:.1f}-{clearance_bins[clear_idx+1]:.1f}"
        print(f"  Speed {speed_range} m/s, Clearance {clear_range} m")


def main():
    parser = argparse.ArgumentParser(description='Visualize MAP-Elites archive')
    parser.add_argument('archive_path', type=str, help='Path to archive JSON file')
    parser.add_argument('--output', type=str, help='Output image path (optional, shows plot if not specified)')
    parser.add_argument('--stats', action='store_true', help='Print statistics instead of visualizing')
    args = parser.parse_args()

    if args.stats:
        print_archive_stats(args.archive_path)
    else:
        visualize_archive(args.archive_path, args.output)


if __name__ == '__main__':
    main()
