import os
import glob
import numpy as np
import time
from helper import R_to_angle  # Updated import
from params import par  # Updated import
from torchvision import transforms
from PIL import Image
import torch
import math

def clean_unused_images():
    """Remove unused image files from sequences/{seq}/image_02/."""
    seq_frame = {
        '00': ['000', '004540'], '01': ['000', '001100'], '02': ['000', '004660'], '03': ['000', '000800'],
        '04': ['000', '000270'], '05': ['000', '002760'], '06': ['000', '001100'], '07': ['000', '001100'],
        '08': ['001100', '005170'], '09': ['000', '001590'], '10': ['000', '001200']
    }
    base_dir = par.data_dir + '/sequences/'
    for dir_id, img_ids in seq_frame.items():
        img_dir_path = f'{base_dir}{dir_id}/image_02/'
        if not os.path.exists(img_dir_path):
            print(f"Image directory {img_dir_path} does not exist, skipping...")
            continue

        print(f'Cleaning images in {dir_id} directory')
        start, end = int(img_ids[0]), int(img_ids[1])
        print(f"Valid range for {dir_id}: {start} to {end}")

        # Clean before start (typically 0, so often no action)
        for idx in range(0, start):
            img_name = f'{idx:010d}.png'
            img_path = f'{img_dir_path}{img_name}'
            if os.path.isfile(img_path):
                print(f"Deleting image {img_path}")
                os.remove(img_path)

        # Clean after end
        for idx in range(end + 1, end + 100):  # Adjust range as needed
            img_name = f'{idx:010d}.png'
            img_path = f'{img_dir_path}{img_name}'
            if os.path.isfile(img_path):
                print(f"Deleting image {img_path}")
                os.remove(img_path)
            else:
                print(f"Image {img_path} not found")

def clean_unused_lidar():
    """Remove unused LiDAR files from sequences/{seq}/velodyne/."""
    seq_frame = {
        '00': ['000', '004540'], '01': ['000', '001100'], '02': ['000', '004660'], '03': ['000', '000800'],
        '04': ['000', '000270'], '05': ['000', '002760'], '06': ['000', '001100'], '07': ['000', '001100'],
        '08': ['001100', '005170'], '09': ['000', '001590'], '10': ['000', '001200']
    }
    base_dir = par.data_dir + '/sequences/'
    for dir_id, img_ids in seq_frame.items():
        lidar_dir_path = f'{base_dir}{dir_id}/velodyne/'
        if not os.path.exists(lidar_dir_path):
            print(f"LiDAR directory {lidar_dir_path} does not exist, skipping...")
            continue

        print(f'Cleaning LiDAR in {dir_id} directory')
        start, end = int(img_ids[0]), int(img_ids[1])
        print(f"Valid range for {dir_id}: {start} to {end}")

        # Clean before start
        for idx in range(0, start):
            lidar_name = f'{idx:010d}.bin'
            lidar_path = f'{lidar_dir_path}{lidar_name}'
            if os.path.isfile(lidar_path):
                print(f"Deleting LiDAR {lidar_path}")
                os.remove(lidar_path)

        # Clean after end
        for idx in range(end + 1, end + 100):  # Match image cleaning range
            lidar_name = f'{idx:010d}.bin'
            lidar_path = f'{lidar_dir_path}{lidar_name}'
            if os.path.isfile(lidar_path):
                print(f"Deleting LiDAR {lidar_path}")
                os.remove(lidar_path)
            else:
                print(f"LiDAR {lidar_path} not found")

def check_file_counts():
    """Count files in image_02 and velodyne directories for each sequence."""
    seq_frame = {
        '00': ['000', '004540'], '01': ['000', '001100'], '02': ['000', '004660'], '03': ['000', '000800'],
        '04': ['000', '000270'], '05': ['000', '002760'], '06': ['000', '001100'], '07': ['000', '001100'],
        '08': ['001100', '005170'], '09': ['000', '001590'], '10': ['000', '001200']
    }
    base_dir = par.data_dir + '/sequences/'
    print("\n=== File Count Check ===")
    for dir_id in seq_frame.keys():
        img_dir_path = f'{base_dir}{dir_id}/image_02/'
        lidar_dir_path = f'{base_dir}{dir_id}/velodyne/'
        
        img_count = len(glob.glob(f'{img_dir_path}*.png')) if os.path.exists(img_dir_path) else 0
        lidar_count = len(glob.glob(f'{lidar_dir_path}*.bin')) if os.path.exists(lidar_dir_path) else 0
        
        print(f"Sequence {dir_id}:")
        print(f"  Images (image_02): {img_count} files")
        print(f"  LiDAR (velodyne): {lidar_count} files")
        if img_count != lidar_count:
            print(f"  WARNING: Mismatch detected! Images: {img_count}, LiDAR: {lidar_count}")

def create_pose_data():
    info = {
        '00': [0, 4540], '01': [0, 1100], '02': [0, 4660], '03': [0, 800], '04': [0, 270], '05': [0, 2760],
        '06': [0, 1100], '07': [0, 1100], '08': [1100, 5170], '09': [0, 1590], '10': [0, 1200]
    }
    start_t = time.time()
    for video in info.keys():
        fn = f'{par.pose_dir}{video}.txt'
        print(f'Transforming {fn}...')
        with open(fn) as f:
            lines = [line.split('\n')[0] for line in f.readlines()]
            poses = [R_to_angle([float(value) for value in l.split(' ')]) for l in lines]
            poses = np.array(poses)
            base_fn = os.path.splitext(fn)[0]
            np.save(base_fn + '.npy', poses)
            print(f'Video {video}: shape={poses.shape}')
    print(f'elapsed time = {time.time() - start_t}')

def calculate_rgb_mean_std(image_path_list, minus_point_5=False):
    n_images = len(image_path_list)
    if n_images == 0:
        raise ValueError("No images found in the provided path list.")
    
    cnt_pixels = 0
    print(f'Numbers of frames in training dataset: {n_images}')
    mean_np = [0, 0, 0]
    mean_tensor = [0, 0, 0]
    to_tensor = transforms.ToTensor()

    for idx, img_path in enumerate(image_path_list):
        print(f'{idx} / {n_images}', end='\r')
        img_as_img = Image.open(img_path)
        img_as_tensor = to_tensor(img_as_img)
        if minus_point_5:
            img_as_tensor = img_as_tensor - 0.5
        img_as_np = np.array(img_as_img)
        img_as_np = np.rollaxis(img_as_np, 2, 0)
        cnt_pixels += img_as_np.shape[1] * img_as_np.shape[2]
        for c in range(3):
            mean_tensor[c] += float(torch.sum(img_as_tensor[c]))
            mean_np[c] += float(np.sum(img_as_np[c]))
    mean_tensor = [v / cnt_pixels for v in mean_tensor]
    mean_np = [v / cnt_pixels for v in mean_np]
    print('mean_tensor = ', mean_tensor)
    print('mean_np = ', mean_np)

    std_tensor = [0, 0, 0]
    std_np = [0, 0, 0]
    for idx, img_path in enumerate(image_path_list):
        print(f'{idx} / {n_images}', end='\r')
        img_as_img = Image.open(img_path)
        img_as_tensor = to_tensor(img_as_img)
        if minus_point_5:
            img_as_tensor = img_as_tensor - 0.5
        img_as_np = np.array(img_as_img)
        img_as_np = np.rollaxis(img_as_np, 2, 0)
        for c in range(3):
            tmp = (img_as_tensor[c] - mean_tensor[c]) ** 2
            std_tensor[c] += float(torch.sum(tmp))
            tmp = (img_as_np[c] - mean_np[c]) ** 2
            std_np[c] += float(np.sum(tmp))
    std_tensor = [math.sqrt(v / cnt_pixels) for v in std_tensor]
    std_np = [math.sqrt(v / cnt_pixels) for v in std_np]
    print('std_tensor = ', std_tensor)
    print('std_np = ', std_np)

if __name__ == '__main__':
    # Separate calls for cleaning
    #clean_unused_images()  # Clean camera images
    #clean_unused_lidar()   # Clean LiDAR data
    create_pose_data()     # Generate pose .npy files
    
    # Calculate RGB mean/std for remaining images
    train_video = ['00', '02', '08', '09', '06', '04', '10']
    image_path_list = []
    for folder in train_video:
        image_path_list += glob.glob(f'{par.data_dir}/sequences/{folder}/image_02/*.png')
    
   # calculate_rgb_mean_std(image_path_list, minus_point_5=True)
    
    # Final check of file counts
    check_file_counts()