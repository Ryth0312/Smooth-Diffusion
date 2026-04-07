import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import argparse
from typing import List, Optional, Tuple
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.gridspec import GridSpec
except ImportError:
    print("ERROR: matplotlib not installed. Install with: pip install matplotlib")
    sys.exit(1)

try:
    from torchvision import transforms
except ImportError:
    print("ERROR: torchvision not installed. Install with: pip install torchvision")
    sys.exit(1)

# Import DINO feature extractor
from losses.dino_smoothness import DinoFeatureExtractor


def load_sequence_frames(sequence_dir: Path) -> List[Image.Image]:
    """Load all frames from a sequence directory.
    Supports: out_imgs/, start_imgs/, or frame_*.png directly in the pair dir (e.g. cat_cat, cat_dog).
    """
    out_imgs_dir = sequence_dir / 'out_imgs'
    start_imgs_dir = sequence_dir / 'start_imgs'
    if out_imgs_dir.exists():
        frame_files = sorted(out_imgs_dir.glob('*.png'))
    elif start_imgs_dir.exists():
        frame_files = sorted(start_imgs_dir.glob('*.png'))
    else:
        # Frames in pair dir (e.g. frame_00.png, frame_01.png from from_prompts-style runs)
        frame_files = sorted(sequence_dir.glob('frame_*.png'), key=lambda p: int(p.stem.rsplit('_', 1)[-1]))
        if not frame_files:
            raise ValueError(f"No out_imgs/, start_imgs/, or frame_*.png found under {sequence_dir}")
    
    if len(frame_files) == 0:
        raise ValueError(f"No PNG files found in {sequence_dir}")
    
    # Load images
    images = [Image.open(f).convert('RGB') for f in frame_files]
    
    return images


def load_baseline_frames(sequence_dir: Path) -> Optional[List[Image.Image]]:
    """Load baseline (initial) frames from start_imgs/ if present.
    Used for green (current) vs red (baseline) DINO velocity comparison.
    Returns None if start_imgs/ does not exist or is empty.
    """
    start_imgs_dir = sequence_dir / 'start_imgs'
    if not start_imgs_dir.exists():
        return None
    frame_files = sorted(start_imgs_dir.glob('*.png'))
    if len(frame_files) == 0:
        return None
    return [Image.open(f).convert('RGB') for f in frame_files]


def compute_frame_metrics(images: List[Image.Image], device: str = 'cuda') -> dict:
    """
    Returns:
        dict with:
        - 'dino_velocities': DINO feature velocity between consecutive frames
        - 'frame_numbers': Frame indices
    """
    print(f"Computing DINO feature metrics on {device}...")
    
    # Setup DINO feature extractor
    #dino = DinoFeatureExtractor(model_name='facebook/dinov2-base', device=device, layers=[11])
    dino = DinoFeatureExtractor(model_name='facebook/dinov3-vitb16-pretrain-lvd1689m', device=device, layers=[11])
    
    # Extract features for all images
    print("Extracting DINO features...")
    with torch.no_grad():
        features = dino.extract_features(images, pool='mean', l2norm=True)  # [T, D]
    
    print(f"Extracted features: {features.shape}")
    
    # Compute first-order differences (velocity)
    velocities = []
    for i in range(len(features) - 1):
        vel = torch.mean((features[i+1] - features[i]) ** 2).item()
        velocities.append(vel)
    
    return {
        'dino_velocities': np.array(velocities),
        'frame_numbers': np.arange(len(images) - 1)
    }


def create_video_from_frames(
    images: List[Image.Image],
    output_path: Path,
    fps: int = 10,
    loop: int = 0
) -> None:
    """
    Args:
        images: List of PIL Images
        output_path: Output video path (.mp4 or .gif)
        fps: Frames per second
        loop: Number of loops (0 = infinite for GIF)
    """
    print(f"Creating video: {output_path}")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('off')
    
    # Display first frame
    img_display = ax.imshow(np.array(images[0]))
    
    def update(frame_idx):
        img_display.set_array(np.array(images[frame_idx]))
        ax.set_title(f'Frame {frame_idx}/{len(images)-1}', fontsize=16)
        return [img_display]
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, update, frames=len(images),
        interval=1000/fps, blit=True, repeat=True
    )
    
    # Save
    if output_path.suffix == '.gif':
        anim.save(str(output_path), writer='pillow', fps=fps)
    else:
        anim.save(str(output_path), writer='ffmpeg', fps=fps, bitrate=2000)
    
    plt.close()
    print(f"Video saved: {output_path}")


def create_metrics_animation(
    metrics: dict,
    output_path: Path,
    fps: int = 10
) -> None:
    """
    Green solid = current DINO velocities; red dashed = initial baseline (if start_imgs available).
    Gray vertical line = current frame indicator.
    """
    print(f"Creating DINO metrics animation: {output_path}")
    
    dino_vels = metrics['dino_velocities']
    frame_nums = metrics['frame_numbers']
    initial_vels = metrics.get('initial_velocities')
    has_baseline = initial_vels is not None and len(initial_vels) == len(dino_vels)
    
    y_max = max(dino_vels) * 1.2 if max(dino_vels) > 0 else 1e-6
    if has_baseline:
        y_max = max(y_max, max(initial_vels) * 1.2)
    
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
    
    # Baseline (red dashed) – full curve, static
    if has_baseline:
        ax1.plot(frame_nums, initial_vels, color='red', linestyle='--', linewidth=1.5,
                 marker='o', markersize=4, label='initial')
    # Current (green solid) – animated
    line_current, = ax1.plot([], [], color='green', linestyle='-', linewidth=1.5,
                             marker='o', markersize=6, label='current')
    ax1.set_xlim(-0.5, len(frame_nums) + 0.5)
    ax1.set_ylim(0, y_max)
    ax1.set_xlabel('Frame Transition', fontsize=12)
    ax1.set_ylabel('DINO Feature Velocity', fontsize=12)
    ax1.set_title('DINO Feature Smoothness (First-Order)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # Current frame indicator (gray vertical line)
    vline1 = ax1.axvline(x=-1, color='gray', linestyle=':', linewidth=1.2, alpha=0.8)
    
    plt.tight_layout()
    
    def update(frame_idx):
        line_current.set_data(frame_nums[:frame_idx+1], dino_vels[:frame_idx+1])
        vline1.set_xdata([frame_idx])
        return [line_current, vline1]
    
    anim = animation.FuncAnimation(
        fig, update, frames=len(frame_nums),
        interval=1000/fps, blit=True, repeat=True
    )
    
    if output_path.suffix == '.gif':
        anim.save(str(output_path), writer='pillow', fps=fps)
    else:
        anim.save(str(output_path), writer='ffmpeg', fps=fps, bitrate=2000)
    
    plt.close()
    print(f"DINO metrics animation saved: {output_path}")


def create_combined_visualization(
    images: List[Image.Image],
    metrics: dict,
    output_path: Path,
    fps: int = 10
) -> None:
    """
    Side-by-side: video on left, DINO metrics on right.
    Green solid = current; red dashed = initial baseline (if start_imgs available).
    Gray vertical line = current frame indicator.
    """
    print(f"Creating combined visualization with DINO metrics: {output_path}")
    
    dino_vels = metrics['dino_velocities']
    frame_nums = metrics['frame_numbers']
    initial_vels = metrics.get('initial_velocities')
    has_baseline = initial_vels is not None and len(initial_vels) == len(dino_vels)
    
    y_max = max(dino_vels) * 1.2 if max(dino_vels) > 0 else 1e-6
    if has_baseline:
        y_max = max(y_max, max(initial_vels) * 1.2)
    
    fig = plt.figure(figsize=(16, 6))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1], height_ratios=[1, 1])
    
    # Left: Image display
    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.axis('off')
    img_display = ax_img.imshow(np.array(images[0]))
    title_img = ax_img.set_title(f'Frame 0/{len(images)-1}', fontsize=14)
    
    # Right: DINO Velocity (green current, red baseline when available)
    ax_vel = fig.add_subplot(gs[:, 1])
    if has_baseline:
        ax_vel.plot(frame_nums, initial_vels, color='red', linestyle='--', linewidth=1.5,
                    marker='o', markersize=4, label='initial')
    line_vel, = ax_vel.plot([], [], color='green', linestyle='-', linewidth=1.5,
                           marker='o', markersize=4, label='current')
    ax_vel.set_xlim(-0.5, len(frame_nums) + 0.5)
    ax_vel.set_ylim(0, y_max)
    ax_vel.set_ylabel('DINO Velocity', fontsize=11)
    ax_vel.set_title('Feature Smoothness (First-Order)', fontsize=12)
    ax_vel.grid(True, alpha=0.3)
    ax_vel.legend(loc='upper right')
    vline_vel = ax_vel.axvline(x=-1, color='gray', linestyle=':', linewidth=1.2, alpha=0.8)
    
    plt.tight_layout()
    
    def update(frame_idx):
        img_display.set_array(np.array(images[frame_idx]))
        title_img.set_text(f'Frame {frame_idx}/{len(images)-1}')
        if frame_idx > 0:
            line_vel.set_data(frame_nums[:frame_idx], dino_vels[:frame_idx])
            vline_vel.set_xdata([frame_idx - 1])
        return [img_display, line_vel, vline_vel]
    
    anim = animation.FuncAnimation(
        fig, update, frames=len(images),
        interval=1000/fps, blit=True, repeat=True
    )
    anim.save(str(output_path), writer='ffmpeg', fps=fps, bitrate=3000)
    plt.close()
    print(f"Combined visualization with DINO metrics saved: {output_path}")


def process_sequence(
    sequence_dir: Path,
    output_dir: Path,
    fps: int = 10,
    device: str = 'cuda',
    create_video: bool = True,
    create_metrics_anim: bool = True,
    create_combined: bool = True
) -> None:
    """
    Process a single interpolation sequence.
    
    Args:
        sequence_dir: Directory containing out_imgs/
        output_dir: Where to save visualizations
        fps: Frames per second for videos
        device: Device for metric computation
        create_video: Whether to create simple video
        create_metrics_anim: Whether to create metrics animation
        create_combined: Whether to create combined visualization
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pair_id = sequence_dir.name
    print(f"\n{'='*60}")
    print(f"Processing: {pair_id}")
    print(f"{'='*60}")
    
    # Load frames (current = out_imgs or main sequence)
    print("Loading frames...")
    images = load_sequence_frames(sequence_dir)
    print(f"✓ Loaded {len(images)} frames")
    
    # Compute metrics (current DINO velocities)
    metrics = compute_frame_metrics(images, device=device)
    print(f"✓ Computed metrics for {len(metrics['dino_velocities'])} transitions")
    
    # Optional: load baseline (start_imgs) for green vs red comparison
    baseline_frames = load_baseline_frames(sequence_dir)
    if baseline_frames is not None and len(baseline_frames) == len(images):
        baseline_metrics = compute_frame_metrics(baseline_frames, device=device)
        metrics['initial_velocities'] = baseline_metrics['dino_velocities']
        print(f"✓ Loaded baseline (start_imgs) for green vs red DINO comparison")
    else:
        metrics['initial_velocities'] = None
    
    # Create visualizations
    if create_video:
        video_path = output_dir / f'{pair_id}_sequence.mp4'
        create_video_from_frames(images, video_path, fps=fps)
    
    if create_metrics_anim:
        metrics_path = output_dir / f'{pair_id}_metrics.mp4'
        create_metrics_animation(metrics, metrics_path, fps=fps)
    
    if create_combined:
        combined_path = output_dir / f'{pair_id}_combined.mp4'
        create_combined_visualization(images, metrics, combined_path, fps=fps)
    
    print(f"All visualizations created for {pair_id}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize interpolation sequences with animated metrics'
    )
    parser.add_argument('--root', type=str, required=True,
                       help='Root directory (e.g., runs/geodesicdiff)')
    parser.add_argument('--pair_id', type=str, nargs='*', default=None,
                       help='Pair ID(s) to process (e.g., cat_cat cat_cat06 cat_dog). If not specified, processes all pairs.')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: <root>/visualizations)')
    parser.add_argument('--fps', type=int, default=10,
                       help='Frames per second for videos (default: 10)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device for metric computation (cuda or cpu)')
    parser.add_argument('--video-only', action='store_true',
                       help='Only create simple video (faster)')
    parser.add_argument('--metrics-only', action='store_true',
                       help='Only create metrics animation (no video)')
    parser.add_argument('--combined-only', action='store_true',
                       help='Only create combined visualization')
    
    args = parser.parse_args()
    
    root_dir = Path(args.root)
    if not root_dir.exists():
        print(f"ERROR: Root directory not found: {root_dir}")
        return 1
    
    # Output directory
    if args.output_dir:
        output_base = Path(args.output_dir)
    else:
        output_base = root_dir / 'visualizations'
    
    # Check device
    if args.device == 'cuda':
        if torch.cuda.is_available():
            print(f"Using device: cuda (GPU: {torch.cuda.get_device_name(0)})")
        else:
            print("Warning: CUDA not available, falling back to CPU")
            args.device = 'cpu'
    else:
        print("Using device: cpu")
    
    # Determine what to create
    if args.video_only:
        create_video, create_metrics, create_combined = True, False, False
    elif args.metrics_only:
        create_video, create_metrics, create_combined = False, True, False
    elif args.combined_only:
        create_video, create_metrics, create_combined = False, False, True
    else:
        # Default: create all
        create_video, create_metrics, create_combined = True, True, True
    
    print("="*60)
    print("Interpolation Visualization")
    print("="*60)
    print(f"Root:        {root_dir}")
    print(f"Output:      {output_base}")
    print(f"FPS:         {args.fps}")
    print(f"Device:      {args.device}")
    print(f"Create video:    {create_video}")
    print(f"Create metrics:  {create_metrics}")
    print(f"Create combined: {create_combined}")
    print("="*60)
    
    # Process sequences
    if args.pair_id:
        # Process only the given pair ID(s)
        pair_ids = list(args.pair_id)
        for i, pid in enumerate(pair_ids, 1):
            sequence_dir = root_dir / pid
            if not sequence_dir.exists():
                print(f"ERROR: Pair directory not found: {sequence_dir}")
                return 1
            if not sequence_dir.is_dir():
                print(f"ERROR: Not a directory: {sequence_dir}")
                return 1
            print(f"\n[{i}/{len(pair_ids)}]")
            output_dir = output_base / pid
            try:
                process_sequence(
                    sequence_dir, output_dir, args.fps, args.device,
                    create_video, create_metrics, create_combined
                )
            except Exception as e:
                print(f"ERROR processing {pid}: {e}")
                import traceback
                traceback.print_exc()
                return 1
    else:
        # Process all pairs
        has_frames = lambda d: (
            (d / 'out_imgs').exists() or (d / 'start_imgs').exists()
            or len(list(d.glob('frame_*.png'))) > 0
        )
        pair_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir() and has_frames(d)])
        
        if len(pair_dirs) == 0:
            print(f"ERROR: No pair directories found in {root_dir} (expected subdirs with out_imgs/, start_imgs/, or frame_*.png)")
            return 1
        
        print(f"\nFound {len(pair_dirs)} pairs to process")
        
        for i, sequence_dir in enumerate(pair_dirs, 1):
            print(f"\n[{i}/{len(pair_dirs)}]")
            try:
                output_dir = output_base / sequence_dir.name
                process_sequence(
                    sequence_dir, output_dir, args.fps, args.device,
                    create_video, create_metrics, create_combined
                )
            except Exception as e:
                print(f"ERROR processing {sequence_dir.name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print("\n" + "="*60)
    print("All visualizations complete!")
    print(f"Results saved to: {output_base}")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    exit(main())
