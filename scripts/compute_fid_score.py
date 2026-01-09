#!/usr/bin/env python3
"""
Compute FID score between generated images and real dataset
"""

import argparse
from pathlib import Path
import sys

try:
    from cleanfid import fid
except ImportError:
    print("Installing clean-fid...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "clean-fid"])
    from cleanfid import fid


def count_images(directory):
    """Count images in directory"""
    extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    count = 0
    for ext in extensions:
        count += len(list(Path(directory).rglob(f'*{ext}')))
    return count


def main():
    parser = argparse.ArgumentParser(description="Compute FID score")
    parser.add_argument("--real_dir", type=str, required=True,
                       help="Directory with real images (e.g., dataset_original/PlantVillage)")
    parser.add_argument("--generated_dir", type=str, required=True,
                       help="Directory with generated images")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for feature extraction")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    args = parser.parse_args()
    
    real_dir = Path(args.real_dir)
    gen_dir = Path(args.generated_dir)
    
    print("=" * 80)
    print("FID Score Computation")
    print("=" * 80)
    print(f"Real images: {real_dir}")
    print(f"Generated images: {gen_dir}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print()
    
    # Validate directories
    if not real_dir.exists():
        print(f"Error: Real directory not found: {real_dir}")
        sys.exit(1)
    if not gen_dir.exists():
        print(f"Error: Generated directory not found: {gen_dir}")
        sys.exit(1)
    
    # Count images
    real_count = count_images(real_dir)
    gen_count = count_images(gen_dir)
    
    print(f"Real images found: {real_count}")
    print(f"Generated images found: {gen_count}")
    print()
    
    if real_count == 0:
        print("Error: No real images found!")
        sys.exit(1)
    if gen_count == 0:
        print("Error: No generated images found!")
        sys.exit(1)
    
    # Compute FID
    print("Computing FID score (this may take a while)...")
    print("Extracting features from real images...")
    
    try:
        fid_score = fid.compute_fid(
            str(real_dir),
            str(gen_dir),
            mode="clean",
            num_workers=4,
            batch_size=args.batch_size,
            device=args.device,
        )
        
        print()
        print("=" * 80)
        print(f"✓ FID Score: {fid_score:.4f}")
        print("=" * 80)
        print()
        print("Interpretation:")
        print("  - Lower FID = Better quality")
        print("  - FID < 50: Excellent")
        print("  - FID 50-100: Good")
        print("  - FID 100-200: Fair")
        print("  - FID > 200: Poor")
        print()
        
        # Save result
        result_file = gen_dir / "fid_score.txt"
        with open(result_file, "w") as f:
            f.write(f"FID Score: {fid_score:.4f}\n")
            f.write(f"Real images: {real_count}\n")
            f.write(f"Generated images: {gen_count}\n")
            f.write(f"Real directory: {real_dir}\n")
            f.write(f"Generated directory: {gen_dir}\n")
        
        print(f"Result saved to: {result_file}")
        
        return fid_score
        
    except Exception as e:
        print(f"Error computing FID: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
