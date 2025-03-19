import os
import shutil

def rename_directories(base_dir, folder_pairs):
    """
    Rename directories in all sequences under base_dir from old_name to new_name.
    
    Args:
        base_dir (str): Base directory containing sequence folders (e.g., "/home/krkavinda/Datasets/KITTI_raw/kitti_data/sequences").
        folder_pairs (list of tuples): List of (old_name, new_name) pairs (e.g., [("image_2", "image_02"), ("image_3", "image_03")]).
    """
    # Ensure base directory exists
    if not os.path.exists(base_dir):
        print(f"Base directory {base_dir} does not exist!")
        return
    
    # List all sequence folders (assuming 00 to 09)
    sequences = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit() and int(d) <= 10])
    
    # Process each sequence
    for seq in sequences:
        for old_name, new_name in folder_pairs:
            old_dir = os.path.join(base_dir, seq, old_name)
            new_dir = os.path.join(base_dir, seq, new_name)
            
            if os.path.exists(old_dir):
                try:
                    shutil.move(old_dir, new_dir)
                    print(f"Renamed: {old_dir} -> {new_dir}")
                except Exception as e:
                    print(f"Failed to rename {old_dir} to {new_dir}: {e}")
            else:
                print(f"Directory {old_dir} not found, skipping...")
    
    print(f"Directory renaming complete. Processed {len(sequences)} sequences.")

# Specific usage for your dataset
if __name__ == "__main__":
    base_dir = "/home/krkavinda/Datasets/KITTI_raw/kitti_data/sequences"
    folder_pairs = [("image_2", "image_02"), ("image_3", "image_03")]
    rename_directories(base_dir, folder_pairs)