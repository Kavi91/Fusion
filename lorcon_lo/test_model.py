import os
import numpy as np
import torch
from model import LoRCoNLO
from utils.process_data import LoRCoNLODataset
from torch.utils.data import DataLoader

# Set dataset root path
root_data_path = "/home/kavi/LoRCoN-LO/data/KITTI/preprocessed_data/"

# Check available sequences (e.g., 00, 01, ..., 10)
available_sequences = sorted([d for d in os.listdir(root_data_path) if d.isdigit()])
if not available_sequences:
    raise ValueError(f"❌ No valid sequences found in {root_data_path}")

print(f"🔹 Found Sequences: {available_sequences}")

# Define batch size & sequence length
batch_size = 10
seq_len = 5

# Select the first available sequence (e.g., "00")
test_sequence = available_sequences[0]  
data_path = os.path.join(root_data_path, test_sequence)

print(f"🔹 Using Sequence: {test_sequence} ({data_path})")

# Load pose file
pose_file = f"/home/kavi/LoRCoN-LO/data/KITTI/pose/{test_sequence}.txt"

if not os.path.exists(pose_file):
    raise FileNotFoundError(f"❌ Pose file not found: {pose_file}")

# Load poses as NumPy array (fix indexing issue)
Y_data = np.loadtxt(pose_file, dtype=str)

# Ensure at least 5 valid samples exist
if len(Y_data) < seq_len:
    raise ValueError(f"❌ Not enough pose data in {pose_file} (found {len(Y_data)} samples)")

# Create dataset
dataset = LoRCoNLODataset(
    img_dir=data_path, 
    Y_data=Y_data,  # ✅ Ensuring Y_data is a NumPy array
    data_idx=np.arange(len(Y_data) - seq_len),  # ✅ Generate valid indices
    seq_sizes={0: seq_len},  # ✅ Ensure sequence sizes are valid
    rnn_size=seq_len, 
    width=900, 
    height=64, 
    depth_name="depth", 
    intensity_name="intensity", 
    normal_name="normal",
    dni_size=2,
    normal_size=6
)

# Create DataLoader
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Load LoRCoN-LO model
print("🔹 Loading LoRCoN-LO Model...")
model = LoRCoNLO(batch_size=batch_size).to("cuda:0")
model.eval()

# Fetch one batch from dataset
print("🔹 Fetching a Batch from Dataset...")
for batch in data_loader:
    dummy_input, _ = batch  # Extract LiDAR input
    dummy_input = dummy_input.to("cuda:0")
    break  # Use one batch for testing

# Forward pass through CNN and RNN
with torch.no_grad():
    print(f"\n🔹 Real Input Shape: {dummy_input.shape}")

    # Reshape input for CNN
    reshaped_input = dummy_input.view(batch_size * seq_len, 10, 64, 900)

    print(f"\n🔹 Reshaped Input to CNN: {reshaped_input.shape}")

    # Pass through CNN
    cnn_features = model.encode_image(reshaped_input)

    print(f"\n✅ CNN Output Shape (Before RNN): {cnn_features.shape}")

    # Flatten CNN Output for RNN
    batch_size_new, channels, height, width = cnn_features.shape
    rnn_input_size = channels * height * width
    rnn_input = cnn_features.view(batch_size, seq_len, rnn_input_size)

    print(f"\n✅ RNN Input Shape: {rnn_input.shape}")

    # Check RNN input size match
    expected_rnn_input_size = model.rnn.input_size
    if rnn_input_size != expected_rnn_input_size:
        print(f"\n⚠️ Mismatch: RNN expected input size {expected_rnn_input_size}, but got {rnn_input_size}!")
    else:
        print(f"\n✅ RNN Input Size Matches Expected Value!")

    # Forward pass through RNN
    try:
        rnn_output, _ = model.rnn(rnn_input)
        print(f"\n✅ Final RNN Output: {rnn_output.shape}")

        # Fully connected layer
        rnn_output = rnn_output.reshape(batch_size * seq_len, -1)
        final_output = model.fc_part(rnn_output)
        final_output = final_output.reshape(batch_size, seq_len, -1)

        print(f"\n✅ Final Pose Estimation Output: {final_output.shape}")

    except RuntimeError as e:
        print(f"\n❌ RNN Forward Pass Failed: {str(e)}")

