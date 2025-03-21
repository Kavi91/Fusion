import os
import glob
import numpy as np
from helper import R_to_angle
from params import par
from torchvision import transforms
from PIL import Image
import torch
import math

def clean_unused_images():
    seq_frame = {
        '00': ['000', '004540'], '01': ['000', '001100'], '02': ['000', '004660'], '03': ['000', '000800'],
        '04': ['000', '000270'], '05': ['000', '002760'], '06': ['000', '001100'], '07': ['000', '001100'],
        '08': ['001100', '005170'], '09': ['000', '001590'], '10': ['000', '001200']
    }
    base_dir = par.data_dir + '/sequences/'
    for dir_id, img_ids in seq_frame.items():
        img_dir_path = f'{base_dir}{dir_id}/image_02/'
        if not os.path.exists(img_dir_path):
            continue

        start, end = int(img_ids[0]), int(img_ids[1])
        for idx in range(0, start):
            img_path = f'{img_dir_path}{idx:010d}.png'
            if os.path.isfile(img_path):
                os.remove(img_path)

        for idx in range(end + 1, end + 100):
            img_path = f'{img_dir_path}{idx:010d}.png'
            if os.path.isfile(img_path):
                os.remove(img_path)

def clean_unused_lidar():
    seq_frame = {
        '00': ['000', '004540'], '01': ['000', '001100'], '02': ['000', '004660'], '03': ['000', '000800'],
        '04': ['000', '000270'], '05': ['000', '002760'], '06': ['000', '001100'], '07': ['000', '001100'],
        '08': ['001100', '005170'], '09': ['000', '001590'], '10': ['000', '001200']
    }
    base_dir = par.data_dir + '/sequences/'
    for dir_id, img_ids in seq_frame.items():
        lidar_dir_path = f'{base_dir}{dir_id}/velodyne/'
        if not os.path.exists(lidar_dir_path):
            continue

        start, end = int(img_ids[0]), int(img_ids[1])
        for idx in range(0, start):
            lidar_path = f'{lidar_dir_path}{idx:010d}.bin'
            if os.path.isfile(lidar_path):
                os.remove(lidar_path)

        for idx in range(end + 1, end + 100):
            lidar_path = f'{lidar_dir_path}{idx:010d}.bin'
            if os.path.isfile(lidar_path):
                os.remove(lidar_path)

def check_file_counts():
    seq_frame = {
        '00': ['000', '004540'], '01': ['000', '001100'], '02': ['000', '004660'], '03': ['000', '000800'],
        '04': ['000', '000270'], '05': ['000', '002760'], '06': ['000', '001100'], '07': ['000', '001100'],
        '08': ['000', '004070'], '09': ['000', '001590'], '10': ['000', '001200']
    }
    base_dir = '/home/krkavinda/Datasets/KITTI_raw/kitti_data/sequences/'
    lidar_base_dir = '/home/krkavinda/Datasets/KITTI_raw/kitti_data/scan/'
    preprocessed_base_dir = '/home/krkavinda/Datasets/KITTI_raw/kitti_data/preprocessed_data/'

    for seq_id in seq_frame.keys():
        img_dir_path = f'{base_dir}{seq_id}/image_02/'
        lidar_dir_path = f'{lidar_base_dir}{seq_id}/velodyne/'
        depth_dir_path = f'{preprocessed_base_dir}{seq_id}/depth/'
        intensity_dir_path = f'{preprocessed_base_dir}{seq_id}/intensity/'
        normal_dir_path = f'{preprocessed_base_dir}{seq_id}/normal/'
        
        img_count = len(glob.glob(f'{img_dir_path}*.png')) if os.path.exists(img_dir_path) else 0
        
        start_frame, end_frame = seq_frame[seq_id]
        start_idx = int(start_frame)
        end_idx = int(end_frame) + 1
        
        lidar_count = 0
        if os.path.exists(lidar_dir_path):
            for frame_idx in range(start_idx, end_idx):
                lidar_file = f'{lidar_dir_path}{frame_idx:06d}.bin'
                if os.path.exists(lidar_file):
                    lidar_count += 1
        
        depth_count = len(glob.glob(f'{depth_dir_path}*.npy')) if os.path.exists(depth_dir_path) else 0
        intensity_count = len(glob.glob(f'{intensity_dir_path}*.npy')) if os.path.exists(intensity_dir_path) else 0
        normal_count = len(glob.glob(f'{normal_dir_path}*.npy')) if os.path.exists(normal_dir_path) else 0
        
        print(f"Sequence {seq_id}:")
        print(f"  Images (image_02): {img_count} files")
        print(f"  LiDAR (velodyne): {lidar_count} files")
        print(f"  Depth (preprocessed): {depth_count} files")
        print(f"  Intensity (preprocessed): {intensity_count} files")
        print(f"  Normal (preprocessed): {normal_count} files")
        
        if img_count != lidar_count:
            print(f"  WARNING: Mismatch detected! Images: {img_count}, LiDAR: {lidar_count}")
        if img_count != depth_count:
            print(f"  WARNING: Mismatch detected! Images: {img_count}, Depth: {depth_count}")
        if img_count != intensity_count:
            print(f"  WARNING: Mismatch detected! Images: {img_count}, Intensity: {intensity_count}")
        if img_count != normal_count:
            print(f"  WARNING: Mismatch detected! Images: {img_count}, Normal: {normal_count}")

def create_pose_data():
    info = {
        '00': [0, 4540], '01': [0, 1100], '02': [0, 4660], '03': [0, 800], '04': [0, 270], '05': [0, 2760],
        '06': [0, 1100], '07': [0, 1100], '08': [1100, 5170], '09': [0, 1590], '10': [0, 1200]
    }
    for video in info.keys():
        fn = f'{par.pose_dir}{video}.txt'
        with open(fn) as f:
            lines = [line.split('\n')[0] for line in f.readlines()]
            poses = [R_to_angle([float(value) for value in l.split(' ')]) for l in lines]
            poses = np.array(poses)
            base_fn = os.path.splitext(fn)[0]
            np.save(base_fn + '.npy', poses)

def calculate_rgb_mean_std(image_path_list, minus_point_5=False):
    n_images = len(image_path_list)
    if n_images == 0:
        raise ValueError("No images found in the provided path list.")
    
    cnt_pixels = 0
    mean = torch.zeros(3)
    to_tensor = transforms.ToTensor()

    # First pass: calculate mean
    for img_path in image_path_list:
        
        img = Image.open(img_path)
        img_tensor = to_tensor(img) / 255.0  # Normalize by dividing by 255
        if minus_point_5:
            img_tensor = img_tensor - 0.5
        cnt_pixels += img_tensor.shape[1] * img_tensor.shape[2]
        mean += torch.sum(img_tensor, dim=(1, 2))

    mean = mean / cnt_pixels

    # Second pass: calculate standard deviation
    std = torch.zeros(3)
    for img_path in image_path_list:
        img = Image.open(img_path)
        img_tensor = to_tensor(img) / 255.0  # Normalize by dividing by 255
        if minus_point_5:
            img_tensor = img_tensor - 0.5
        for c in range(3):
            std[c] += torch.sum((img_tensor[c] - mean[c]) ** 2)

    std = torch.sqrt(std / cnt_pixels)

    print(f"Mean (after normalization): {mean.tolist()}")
    print(f"Std (after normalization): {std.tolist()}")

if __name__ == "__main__":
    train_video = ["00", "01", "02", "05", "06", "07", "08"]
    image_path_list = []
    for folder in train_video:
        image_path_list += glob.glob(f'{par.data_dir}/sequences/{folder}/image_02/*.png')
    
    calculate_rgb_mean_std(image_path_list, minus_point_5=True)