#!/usr/bin/env python3
"""
Simplified Chess Piece Detection for YOLOv8 Annotations
Detects single large black or cream chess piece on white background.
"""

import cv2
import numpy as np
import os
from pathlib import Path
import argparse


class ChessPieceDetector:
    def __init__(self, min_area=5000):
        """
        Initialize the chess piece detector.
        Detects single black or cream/beige chess piece on white background.
        
        Args:
            min_area: Minimum contour area to consider as a chess piece (default: 5000)
        """
        self.min_area = min_area
    
    def detect_color_region(self, image, is_dark=True):
        """
        Detect large dark (black) or light (cream/beige) region on white background.
        
        Args:
            image: Input BGR image
            is_dark: If True, detect dark/black regions. If False, detect cream/beige regions.
            
        Returns:
            Binary mask with the detected region
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if is_dark:
            # Detect BLACK pieces
            # Anything darker than threshold is considered black
            _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        else:
            # Detect CREAM/BEIGE pieces
            # Cream pieces are lighter but not pure white (white background)
            # We want: not too dark (> 130) and not too light/white (< 200)
            _, gray_binary = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)
            _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            white_mask = cv2.bitwise_not(white_mask)
            
            # Combine: not too dark, not too white
            binary = cv2.bitwise_and(gray_binary, white_mask)
        
        # Clean up the mask with morphological operations
        kernel = np.ones((15, 15), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Fill holes to get solid piece
        kernel_fill = np.ones((25, 25), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_fill, iterations=3)
        
        return binary
    
    def detect_pieces(self, image):
        """
        Detect single large chess piece (black or cream) on white background.
        
        Args:
            image: Input BGR image
            
        Returns:
            List of bounding boxes [(x, y, w, h), ...] - should contain 1 box
        """
        img_height, img_width = image.shape[:2]
        img_area = img_height * img_width
        
        # Max area should be 80% of image
        max_area = img_area * 0.8
        
        # Try detecting black piece first
        binary_dark = self.detect_color_region(image, is_dark=True)
        
        # Try detecting cream piece
        binary_light = self.detect_color_region(image, is_dark=False)
        
        # Find contours for both
        contours_dark, _ = cv2.findContours(binary_dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_light, _ = cv2.findContours(binary_light, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Combine all contours
        all_contours = list(contours_dark) + list(contours_light)
        
        if not all_contours:
            print("Warning: No regions detected")
            return []
        
        # Find the largest contour (should be the chess piece)
        largest_contour = max(all_contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # Check if area is reasonable
        if area < self.min_area:
            print(f"Warning: Largest region too small (area={area:.0f}, min={self.min_area})")
            return []
        
        if area > max_area:
            print(f"Warning: Largest region too large (area={area:.0f}, max={max_area:.0f})")
            return []
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        return [(x, y, w, h)]
    
    def bbox_to_yolo(self, bbox, img_width, img_height, class_id=0):
        """
        Convert bounding box to YOLOv8 format.
        
        Args:
            bbox: Bounding box (x, y, w, h)
            img_width: Image width
            img_height: Image height
            class_id: Class ID for the object
            
        Returns:
            YOLOv8 format: (class_id, x_center_norm, y_center_norm, width_norm, height_norm)
        """
        x, y, w, h = bbox
        
        # Calculate center coordinates
        x_center = x + w / 2
        y_center = y + h / 2
        
        # Normalize coordinates
        x_center_norm = x_center / img_width
        y_center_norm = y_center / img_height
        width_norm = w / img_width
        height_norm = h / img_height
        
        return (class_id, x_center_norm, y_center_norm, width_norm, height_norm)
    
    def save_yolo_annotation(self, bboxes, img_width, img_height, output_path, class_id=0):
        """
        Save bounding boxes in YOLOv8 annotation format.
        
        Args:
            bboxes: List of bounding boxes [(x, y, w, h), ...]
            img_width: Image width
            img_height: Image height
            output_path: Path to save the annotation file
            class_id: Class ID for all detected objects
        """
        with open(output_path, 'w') as f:
            for bbox in bboxes:
                yolo_bbox = self.bbox_to_yolo(bbox, img_width, img_height, class_id)
                # Format: class_id x_center y_center width height
                f.write(f"{yolo_bbox[0]} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} "
                       f"{yolo_bbox[3]:.6f} {yolo_bbox[4]:.6f}\n")
    
    def visualize_detections(self, image, bboxes, output_path=None):
        """
        Visualize detected bounding boxes on the image.
        
        Args:
            image: Input BGR image
            bboxes: List of bounding boxes [(x, y, w, h), ...]
            output_path: Optional path to save the visualization
            
        Returns:
            Image with drawn bounding boxes
        """
        vis_image = image.copy()
        
        for bbox in bboxes:
            x, y, w, h = bbox
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(vis_image, 'chess_piece', (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        if output_path:
            cv2.imwrite(output_path, vis_image)
        
        return vis_image


def process_single_image(image_path, output_dir, detector, visualize=True):
    """
    Process a single image and generate annotations.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save annotations
        detector: ChessPieceDetector instance
        visualize: Whether to save visualization
    """
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    img_height, img_width = image.shape[:2]
    
    # Detect pieces
    bboxes = detector.detect_pieces(image)
    
    if len(bboxes) == 0:
        print(f"No chess piece detected in {image_path}")
    elif len(bboxes) == 1:
        print(f"✓ Detected 1 chess piece in {image_path}")
    else:
        print(f"Warning: Detected {len(bboxes)} pieces in {image_path} (expected 1)")
    
    # Create output directories
    labels_dir = Path(output_dir) / 'labels'
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Save annotation
    image_name = Path(image_path).stem
    annotation_path = labels_dir / f"{image_name}.txt"
    detector.save_yolo_annotation(bboxes, img_width, img_height, annotation_path)
    print(f"  Saved annotation to {annotation_path}")
    
    # Save visualization
    if visualize:
        vis_dir = Path(output_dir) / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)
        vis_path = vis_dir / f"{image_name}_detected.jpg"
        detector.visualize_detections(image, bboxes, vis_path)
        print(f"  Saved visualization to {vis_path}")


def process_directory(input_dir, output_dir, detector, visualize=True):
    """
    Process all images in a directory.
    
    Args:
        input_dir: Directory containing images
        output_dir: Directory to save annotations
        detector: ChessPieceDetector instance
        visualize: Whether to save visualizations
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    input_path = Path(input_dir)
    
    image_files = [f for f in input_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    print(f"Found {len(image_files)} images in {input_dir}")
    print("=" * 60)
    
    for image_file in image_files:
        print(f"\nProcessing {image_file.name}...")
        process_single_image(image_file, output_dir, detector, visualize)
    
    print("\n" + "=" * 60)
    print(f"✓ Processing complete! Annotations saved to {output_dir}/labels/")
    print(f"✓ Check {output_dir}/visualizations/ to verify detections")


def create_yaml_config(output_dir, class_names=['chess_piece']):
    """
    Create a YOLOv8 dataset configuration YAML file.
    
    Args:
        output_dir: Output directory for the dataset
        class_names: List of class names
    """
    yaml_path = Path(output_dir) / 'dataset.yaml'
    
    yaml_content = f"""# YOLOv8 Dataset Configuration
# Generated by chess_piece_detector.py

path: {Path(output_dir).absolute()}  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')
test: images/test  # test images (optional)

# Classes
names:
"""
    
    for i, class_name in enumerate(class_names):
        yaml_content += f"  {i}: {class_name}\n"
    
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✓ Created YOLOv8 dataset config at {yaml_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Detect single black or cream chess piece on white background and generate YOLOv8 annotations'
    )
    # parser.add_argument('input', help='Input image file or directory')
    parser.add_argument('-o', '--output', default='output',
                       help='Output directory for annotations (default: output)')
    parser.add_argument('--min-area', type=int, default=5000,
                       help='Minimum contour area (default: 5000)')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Skip creating visualization images')
    parser.add_argument('--create-yaml', action='store_true',
                       help='Create YOLOv8 dataset.yaml configuration file')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = ChessPieceDetector(min_area=args.min_area)
    
    # Check if input is file or directory
    input_path = "dataset_yolo/images/test"
    
    # if input_path.is_file():
    #     process_single_image(input_path, args.output, detector, 
    #                        visualize=not args.no_visualize)
    # elif input_path.is_dir():
    process_directory(input_path, args.output, detector, 
                        visualize=not args.no_visualize)
    # else:
    #     print(f"Error: {args.input} is not a valid file or directory")
    #     return
    
    # Create YAML config if requested
    if args.create_yaml:
        create_yaml_config(args.output)


if __name__ == '__main__':
    main()