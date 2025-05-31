import argparse
from pathlib import Path
import torch
import cv2
import numpy as np

from MambaGlue.mambaglue.mambaglue import MambaGlue
from mambaglue.superpoint import SuperPoint
from mambaglue.utils import read_image, numpy_image_to_torch, match_pair
from mambaglue.viz2d import plot_images, plot_matches, save_plot


def parse_args():
    parser = argparse.ArgumentParser(description='MambaGlue Demo: Match a pair of images')
    parser.add_argument('--image1', type=str, required=True, help='Path to first image')
    parser.add_argument('--image2', type=str, required=True, help='Path to second image')
    parser.add_argument('--output', type=str, default='matches.png', help='Path to save visualization')
    parser.add_argument('--max_keypoints', type=int, default=1024, help='Maximum number of keypoints to detect')
    parser.add_argument('--resize', type=int, default=1024, help='Resize image max dimension')
    parser.add_argument('--threshold', type=float, default=0.2, help='Matching threshold')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to use for computation (cuda/cpu)')
    parser.add_argument('--display', action='store_true', help='Display the visualization')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"Running on device: {args.device}")
    
    # Load images
    print(f"Loading images: {args.image1}, {args.image2}")
    image0 = read_image(args.image1)
    image1 = read_image(args.image2)
    
    # Convert images to torch tensors
    image0_torch = numpy_image_to_torch(image0).to(args.device)
    image1_torch = numpy_image_to_torch(image1).to(args.device)
    
    # Initialize feature extractor and matcher
    print("Initializing SuperPoint extractor and MambaGlue matcher...")
    extractor = SuperPoint(max_num_keypoints=args.max_keypoints).to(args.device).eval()
    matcher = MambaGlue(features="superpoint").to(args.device).eval()
    
    # Extract features and match
    print("Extracting features and matching...")
    with torch.no_grad():
        feats0, feats1, matches01 = match_pair(
            extractor, matcher, image0_torch, image1_torch, 
            device=args.device, resize=args.resize
        )
    
    # Get keypoints and matches
    kpts0 = feats0["keypoints"]
    kpts1 = feats1["keypoints"]
    matches = matches01["matches"]
    scores = matches01["scores"]
    
    # Filter matches by threshold
    valid = scores > args.threshold
    matches = matches[valid]
    scores = scores[valid]
    
    print(f"Found {len(matches)} matches with score > {args.threshold}")
    
    # Visualization
    print(f"Visualizing matches and saving to {args.output}")
    color = cm = np.array([0, 1.0, 0])[None]  # Green
    text = [
        f'MambaGlue matches: {len(matches)}',
        f'Image size: {image0.shape[:2]}, {image1.shape[:2]}',
    ]
    
    # Create visualization
    plot_images([image0, image1], titles=text)
    
    # Draw matches
    if len(matches) > 0:
        kpts0_matched = kpts0[matches[:, 0]]
        kpts1_matched = kpts1[matches[:, 1]]
        plot_matches(kpts0_matched, kpts1_matched, color=color, ps=4, lw=1)
    
    # Save visualization
    save_plot(args.output)
    print(f"Visualization saved to {args.output}")
    
    # Display the result if requested
    if args.display:
        print("Opening visualization in browser...")
        import subprocess
        subprocess.run(["$BROWSER", str(Path(args.output).absolute())])


if __name__ == "__main__":
    main()