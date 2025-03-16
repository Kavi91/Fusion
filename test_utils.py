# FUSION/test_utils.py
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from lorcon_lo.utils.process_data import LoRCoNLODataset, count_seq_sizes, process_input_data
from deepvo.data_helper import get_data_info, ImageSequenceDataset
from utils import load_timestamps, create_frame_mapping
import yaml

# Mock LoRCoNLODataset class for testing
class MockLoRCoNLODataset(Dataset):
    def __init__(self, img_dir, Y_data, data_idx, seq_sizes, rnn_size, width, height, depth_name, intensity_name, normal_name, dni_size, normal_size):
        self.img_dir = img_dir
        self.Y_data = Y_data
        self.data_idx = data_idx
        self.seq_sizes = seq_sizes
        self.rnn_size = rnn_size
        self.width = width
        self.height = height
        self.depth_name = depth_name
        self.intensity_name = intensity_name
        self.normal_name = normal_name
        self.dni_size = dni_size
        self.normal_size = normal_size

    def __len__(self):
        return len(self.data_idx)

    def __getitem__(self, idx):
        # Placeholder return; actual implementation not needed for mapping
        return torch.zeros(self.rnn_size, 10, self.height, self.width), torch.zeros(self.rnn_size, 6)

def test_frame_synchronization():
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Load DeepVO DataFrame for sequence 00 with preserved order
    deepvo_df = get_data_info(["00"], [4, 4], 1, 1, shuffle=False, sort=False, config=config)  # Explicitly disable shuffle and sort
    if deepvo_df.empty:
        print("DeepVO DataFrame is empty, check image and pose file paths")
        return

    # Load LoRCoN-LO data for sequence 00
    seq_sizes = count_seq_sizes(config["lorcon_lo"]["preprocessed_folder"], ["00"], {})
    Y_data = process_input_data(config["lorcon_lo"]["preprocessed_folder"], config["lorcon_lo"]["relative_pose_folder"], ["00"], seq_sizes)
    data_idx = np.arange(0, seq_sizes["00"] - config["lorcon_lo"]["rnn_size"] + 1, dtype=int)
    lorcon_data = MockLoRCoNLODataset(
        config["lorcon_lo"]["preprocessed_folder"],
        Y_data,
        data_idx,
        seq_sizes,
        config["lorcon_lo"]["rnn_size"],
        config["lorcon_lo"]["image_width"],
        config["lorcon_lo"]["image_height"],
        config["lorcon_lo"]["depth_name"],
        config["lorcon_lo"]["intensity_name"],
        config["lorcon_lo"]["normal_name"],
        config["lorcon_lo"]["dni_size"],
        config["lorcon_lo"]["normal_size"]
    )

    # Create frame mapping
    frame_pairs = create_frame_mapping(deepvo_df, lorcon_data, config)

    # Print results
    print(f"Number of frame pairs: {len(frame_pairs)}")
    if frame_pairs:
        print("First 5 frame pairs:")
        for pair in frame_pairs[:5]:
            deepvo_idx, lorcon_idx = pair
            video_id = deepvo_df.iloc[deepvo_idx]["image_path"][0].split("/")[-3]
            deepvo_frame = int(os.path.basename(deepvo_df.iloc[deepvo_idx]["image_path"][0]).replace(".png", ""))
            start_id = lorcon_data.data_idx[lorcon_idx]
            image_pre_path_args = lorcon_data.Y_data[start_id, 0].split(" ")
            lorcon_frame = int(image_pre_path_args[1].split(".")[0])
            timestamp_file = os.path.join(config["deepvo"]["timestamp_folder"], f"times_{video_id}.txt")
            timestamps = load_timestamps(timestamp_file)
            print(f"DeepVO idx {deepvo_idx} (frame {deepvo_frame}, time {timestamps[deepvo_frame]}), "
                  f"LoRCoN idx {lorcon_idx} (frame {lorcon_frame}, time {timestamps[lorcon_frame]})")

if __name__ == "__main__":
    test_frame_synchronization()