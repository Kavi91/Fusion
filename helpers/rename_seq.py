import os

# Directory containing the image files
image_dir = "/home/kavi/Datasets/KITTI_raw/kitti_data/sequences/08/image_02/"

# Current starting index (to be subtracted)
current_start_idx = 1100

# Function to rename files to 10-digit format starting from 0000000000
def rename_to_10_digits_from_zero(directory, current_start_idx):
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist!")
        return
    
    # Get all .png files in the directory
    files = sorted([f for f in os.listdir(directory) if f.endswith('.png')])
    
    for old_name in files:
        # Extract the current index from the filename (e.g., "0000001100.png" → 1100)
        old_idx = int(old_name.split('.')[0])
        # Subtract the current starting index to reset to 0 (e.g., 1100 → 0)
        new_idx = old_idx - current_start_idx
        if new_idx < 0:
            print(f"Warning: Index {new_idx} for file {old_name} is negative after subtracting {current_start_idx}. Skipping...")
            continue
        # Convert to 10-digit format (e.g., 0 → "0000000000")
        new_idx_str = f"{new_idx:010d}"
        # New filename (e.g., "0000000000.png")
        new_name = f"{new_idx_str}.png"
        # Old and new file paths
        old_path = os.path.join(directory, old_name)
        new_path = os.path.join(directory, new_name)
        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed {old_path} to {new_path}")

# Run the renaming function
rename_to_10_digits_from_zero(image_dir, current_start_idx)