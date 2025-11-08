import numpy as np

# ------------------------------------------------------------------------
# BOARD STABILITY
# Contains functions to check if the board streaming is stable
# ------------------------------------------------------------------------

# Check if there is random variation in a buffer of the corner points ot check if the board is stable
def calculate_corner_variation(pts_buffer, threshold=10.0):    
    # Filter out None values
    valid_pts = [pts for pts in pts_buffer if pts is not None]
    
    if len(valid_pts) < 2:
        return True, 0.0, {}  # Not enough data, assume stable
    
    # Convert to numpy array
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
