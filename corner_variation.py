import numpy as np

def calculate_corner_variation(pts_buffer, threshold=10.0):
    """
    Check if there is large variation in board corner positions.
    
    Args:
        pts_buffer: List of corner arrays, each shape (4, 2) with format:
                    [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
        threshold: Maximum allowed standard deviation (pixels) to consider stable
    
    Returns:
        tuple: (is_stable, max_std, variation_details)
            - is_stable: True if board is stable (low variation)
            - max_std: Maximum standard deviation across all corners
            - variation_details: Dict with per-corner standard deviations
    """
    # Filter out None values
    valid_pts = [pts for pts in pts_buffer if pts is not None]
    
    if len(valid_pts) < 2:
        return True, 0.0, {}  # Not enough data, assume stable
    
    # Convert to numpy array: shape (n_frames, 4, 2)
    pts_array = np.array(valid_pts)
    
    # Calculate standard deviation for each corner in x and y
    stds = []
    variation_details = {}
    
    for corner_idx in range(4):
        x_std = np.std(pts_array[:, corner_idx, 0])
        y_std = np.std(pts_array[:, corner_idx, 1])
        
        # Use Euclidean distance as combined variation measure
        combined_std = np.sqrt(x_std**2 + y_std**2)
        stds.append(combined_std)
        
        variation_details[f'corner_{corner_idx}'] = {
            'x_std': float(x_std),
            'y_std': float(y_std),
            'combined_std': float(combined_std)
        }
    
    max_std = max(stds)
    is_stable = max_std < threshold
    
    return is_stable, max_std, variation_details
