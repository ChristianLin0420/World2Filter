#!/usr/bin/env python3
"""
Preprocess masks using SAM3 for World2Filter training.

This script generates and caches segmentation masks for all observations
in a dataset, which can then be used for efficient training.

Usage:
    python scripts/preprocess_masks.py --data_dir ./data/episodes --output_dir ./mask_cache
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from tqdm import tqdm

from src.models.segmentation.mask_processor import MaskProcessor
from src.data.replay_buffer import Episode


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess SAM3 masks")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing episode data (*.npz files)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./mask_cache",
        help="Output directory for cached masks",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="robot",
        help="Text prompt for SAM3 segmentation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for SAM3 inference",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--no-sam3",
        action="store_true",
        help="Use fallback segmentation instead of SAM3",
    )
    return parser.parse_args()


def load_episodes(data_dir: Path):
    """Load all episodes from directory."""
    episodes = []
    episode_files = sorted(data_dir.glob("episode_*.npz"))
    
    print(f"Found {len(episode_files)} episode files")
    
    for ep_file in tqdm(episode_files, desc="Loading episodes"):
        data = dict(np.load(ep_file))
        episode = Episode.from_dict(data)
        episodes.append((ep_file, episode))
    
    return episodes


def main():
    args = parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        return 1
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize mask processor
    print("\nInitializing mask processor...")
    processor = MaskProcessor(
        cache_dir=output_dir,
        prompt=args.prompt,
        device=args.device,
        use_sam3=not args.no_sam3,
    )
    
    # Load episodes
    print(f"\nLoading episodes from {data_dir}...")
    episodes = load_episodes(data_dir)
    
    if not episodes:
        print("No episodes found!")
        return 1
    
    # Process each episode
    print(f"\nProcessing {len(episodes)} episodes...")
    
    total_frames = 0
    for ep_file, episode in tqdm(episodes, desc="Processing episodes"):
        # Get masks for all observations in episode
        fg_masks, bg_masks = processor.process_episode(episode.obs)
        
        # Save masks alongside episode
        mask_file = output_dir / f"{ep_file.stem}_masks.npz"
        np.savez_compressed(
            mask_file,
            fg_masks=fg_masks,
            bg_masks=bg_masks,
        )
        
        total_frames += len(episode.obs)
    
    # Print statistics
    print(f"\nProcessing complete!")
    print(f"  Total frames processed: {total_frames}")
    print(f"  Cache statistics: {processor.cache_stats}")
    print(f"  Masks saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

