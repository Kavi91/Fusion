import os
import yaml
from tqdm import tqdm
import gen_data_utils

def gen_spherical_data(scan_folder, rgb_folder, dst_folder, dataset, fov_up, fov_down, proj_H, proj_W, max_range, seq_pbar):
    """Generate spherical data with progress tracking for each type, including RGB."""
    with tqdm(total=4, desc=f"Processing {os.path.basename(scan_folder)}", leave=False, unit="type") as type_pbar:
        #gen_data_utils.gen_spherical_depth_data(scan_folder, rgb_folder, dst_folder, dataset, fov_up=fov_up, fov_down=fov_down, proj_H=proj_H, proj_W=proj_W, max_range=max_range, pbar=type_pbar)
        #type_pbar.update(1)
        #gen_data_utils.gen_spherical_intensity_data(scan_folder, rgb_folder, dst_folder, dataset=dataset, fov_up=fov_up, fov_down=fov_down, proj_H=proj_H, proj_W=proj_W, max_range=max_range, pbar=type_pbar)
        #type_pbar.update(1)
        #gen_data_utils.gen_spherical_normal_data(scan_folder, rgb_folder, dst_folder, dataset, fov_up=fov_up, fov_down=fov_down, proj_H=proj_H, proj_W=proj_W, max_range=max_range, pbar=type_pbar)
        #type_pbar.update(1)
        gen_data_utils.gen_spherical_rgb_data(scan_folder, rgb_folder, dst_folder, dataset, fov_up=fov_up, fov_down=fov_down, proj_H=proj_H, proj_W=proj_W, max_range=max_range, pbar=type_pbar)
        type_pbar.update(1)

if __name__ == "__main__":
    # Load config file from specified path
    config_filename = '/home/kavi/Fusion/config.yaml'
    with open(config_filename, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Extract from lorcon_lo section
    lorcon_config = config["lorcon_lo"]
    preprocessed_folder = lorcon_config["preprocessed_folder"]
    scan_base_folder = lorcon_config["scan_folder"]
    rgb_base_folder = "/home/kavi/Datasets/KITTI_raw/kitti_data/sequences"  # Correct RGB location
    data_seqs = lorcon_config["data_seqs"].split(",")
    dataset = config["dataset"]

    # Set parameters for original resolution
    fov_up = 3.0
    fov_down = -25.0
    proj_H = 64  # Original resolution
    proj_W = 900  # Original resolution
    max_range = 50

    with tqdm(total=len(data_seqs), desc="Processing Sequences", unit="seq") as seq_pbar:
        for seq in data_seqs:
            scan_folder = os.path.join(scan_base_folder, seq, "velodyne")
            rgb_folder = os.path.join(rgb_base_folder, seq, "image_02")  # Specific RGB subfolder
            dst_folder = os.path.join(preprocessed_folder, seq)
            
            os.makedirs(scan_folder, exist_ok=True)
            os.makedirs(rgb_folder, exist_ok=True)
            os.makedirs(dst_folder, exist_ok=True)

            print(f"Checking files in scan_folder: {len([f for f in os.listdir(scan_folder) if f.endswith('.bin')])} .bin files")
            print(f"Checking files in rgb_folder: {len([f for f in os.listdir(rgb_folder) if f.endswith('.png')])} .png files")

            gen_spherical_data(scan_folder, rgb_folder, dst_folder, dataset, fov_up, fov_down, proj_H, proj_W, max_range, seq_pbar)
            
            seq_pbar.set_postfix({"Last Seq": seq})
            seq_pbar.update(1)