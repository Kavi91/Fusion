# FUSION/utils.py
import os
import numpy as np

def load_timestamps(timestamp_file):
    """
    Load timestamps from KITTI timestamps.txt file.
    
    Args:
        timestamp_file (str): Path to timestamps.txt or times_XX.txt.
    
    Returns:
        list: List of timestamps in seconds.
    """
    timestamps = []
    with open(timestamp_file, 'r') as f:
        for line in f:
            timestamp = float(line.strip())
            timestamps.append(timestamp)
    return timestamps

def create_frame_mapping(deepvo_df, lorcon_data, config):
    """
    Create a mapping of synchronized frame indices for DeepVO and LoRCoN-LO.
    
    Args:
        deepvo_df (pd.DataFrame): DeepVO DataFrame.
        lorcon_data (LoRCoNLODataset): LoRCoN-LO dataset.
        config (dict): Configuration dictionary.
    
    Returns:
        list: List of (deepvo_idx, lorcon_idx) pairs.
    """
    # Load timestamps for DeepVO (camera)
    deepvo_timestamps = {}
    for idx, row in deepvo_df.iterrows():
        video_id = row["image_path"][0].split("/")[-3]  # Extract video ID (e.g., "00")
        # Extract the frame ID from the first image in the sequence
        frame_id = int(os.path.basename(row["image_path"][0]).replace(".png", ""))  # Extract frame ID
        timestamp_file = os.path.join(config["deepvo"]["timestamp_folder"], f"times_{video_id}.txt")
        if os.path.exists(timestamp_file):
            timestamps = load_timestamps(timestamp_file)
            if frame_id < len(timestamps):
                deepvo_timestamps[idx] = (video_id, frame_id, timestamps[frame_id])
            else:
                print(f"DeepVO idx {idx}: Frame {frame_id} exceeds timestamp length {len(timestamps)} for video {video_id}")
        else:
            print(f"DeepVO idx {idx}: Timestamp file {timestamp_file} not found")

    # Load timestamps for LoRCoN-LO (LiDAR)
    lorcon_timestamps = {}
    for idx in range(len(lorcon_data)):
        start_id = lorcon_data.data_idx[idx]
        image_pre_path_args = lorcon_data.Y_data[start_id, 0].split(" ")
        video_id = image_pre_path_args[0]  # e.g., "00"
        frame_id = int(image_pre_path_args[1].split(".")[0])  # e.g., 0 for "000000.npy"
        timestamp_file = os.path.join(config["lorcon_lo"]["timestamp_folder"], f"times_{video_id}.txt")
        if os.path.exists(timestamp_file):
            timestamps = load_timestamps(timestamp_file)
            if frame_id < len(timestamps):
                lorcon_timestamps[idx] = (video_id, frame_id, timestamps[frame_id])
            else:
                print(f"LoRCoN idx {idx}: Frame {frame_id} exceeds timestamp length {len(timestamps)} for video {video_id}")
        else:
            print(f"LoRCoN idx {idx}: Timestamp file {timestamp_file} not found")

    # Debug: Print first few timestamps for verification
    if deepvo_timestamps:
        print("DeepVO first 5 timestamps:", list(deepvo_timestamps.values())[:5])
    if lorcon_timestamps:
        print("LoRCoN first 5 timestamps:", list(lorcon_timestamps.values())[:5])

    # Align frames based on timestamps
    frame_pairs = []
    deepvo_indices = sorted(deepvo_timestamps.keys())
    lorcon_indices = sorted(lorcon_timestamps.keys())
    for deepvo_idx in deepvo_indices:
        deepvo_video, deepvo_frame, deepvo_time = deepvo_timestamps[deepvo_idx]
        closest_lorcon_idx = None
        min_time_diff = float('inf')
        for lorcon_idx in lorcon_indices:
            lorcon_video, lorcon_frame, lorcon_time = lorcon_timestamps[lorcon_idx]
            if deepvo_video != lorcon_video:
                continue
            time_diff = abs(deepvo_time - lorcon_time)
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_lorcon_idx = lorcon_idx
        if closest_lorcon_idx is not None and min_time_diff < 0.01:  # 10ms threshold
            frame_pairs.append((deepvo_idx, closest_lorcon_idx))
            print(f"Paired: DeepVO idx {deepvo_idx} (frame {deepvo_frame}, time {deepvo_time}) with LoRCoN idx {closest_lorcon_idx} (frame {lorcon_frame}, time {lorcon_time}), time diff {min_time_diff}")
    
    print(f"Number of frame pairs: {len(frame_pairs)}")
    return frame_pairs