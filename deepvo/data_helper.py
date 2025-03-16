# deepvo/data_helper.py
import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import transforms
import time

# FUSION/deepvo/data_helper.py (excerpt for get_data_info)
def get_data_info(folder_list, seq_len_range, overlap, sample_times=1, pad_y=False, shuffle=False, sort=False, config=None):
    """
    Generate data info dataframe from image and pose data.
    
    Args:
        folder_list (list): List of video folders to process.
        seq_len_range (tuple): Range of sequence lengths (min, max).
        overlap (int): Overlap between sequences.
        sample_times (int): Number of times to sample sequences.
        pad_y (bool): Whether to pad ground truth with zeros.
        shuffle (bool): Whether to shuffle the dataframe (set to False).
        sort (bool): Whether to sort by sequence length (set to False).
        config (dict): Configuration dictionary from config.yaml.
    """
    X_path, Y = [], []
    X_len = []
    for folder in folder_list:
        start_t = time.time()
        poses = np.load(os.path.join(config["deepvo"]["pose_dir"], f"{folder}.npy"))  # (n_images, 6)
        fpaths = glob.glob(os.path.join(config["deepvo"]["image_dir"], f"{folder}/image_02/*.png"))
        fpaths.sort()
        print(f"Video {folder}: {len(fpaths)} frames, {poses.shape[0]} poses")
        if seq_len_range[0] == seq_len_range[1]:
            if sample_times > 1:
                sample_interval = int(np.ceil(seq_len_range[0] / sample_times))
                start_frames = list(range(0, seq_len_range[0], sample_interval))
                print('Sample start from frame {}'.format(start_frames))
            else:
                start_frames = [0]

            for st in start_frames:
                seq_len = seq_len_range[0]
                n_frames = len(fpaths) - st
                jump = seq_len - overlap
                res = n_frames % seq_len
                if res != 0:
                    n_frames = n_frames - res
                x_segs = [fpaths[i:i+seq_len] for i in range(st, n_frames, jump)]
                y_segs = [poses[i:i+seq_len] for i in range(st, n_frames, jump)]
                print(f"Video {folder}: Added {len(x_segs)} sequences of length {seq_len}")
                Y += y_segs
                X_path += x_segs
                X_len += [len(xs) for xs in x_segs]
        else:
            assert(overlap < min(seq_len_range))
            n_frames = len(fpaths)
            min_len, max_len = seq_len_range[0], seq_len_range[1]
            for i in range(sample_times):
                start = 0
                while True:
                    n = np.random.random_integers(min_len, max_len)
                    if start + n < n_frames:
                        x_seg = fpaths[start:start+n] 
                        X_path.append(x_seg)
                        if not pad_y:
                            Y.append(poses[start:start+n])
                        else:
                            pad_zero = np.zeros((max_len-n, 6))
                            padded = np.concatenate((poses[start:start+n], pad_zero))
                            Y.append(padded.tolist())
                    else:
                        print('Last %d frames is not used' %(start+n-n_frames))
                        break
                    start += n - overlap
                    X_len.append(len(x_seg))
        print('Folder {} finish in {} sec'.format(folder, time.time()-start_t))
    
    # Convert to pandas dataframes without sorting or shuffling
    data = {'seq_len': X_len, 'image_path': X_path, 'pose': Y}
    df = pd.DataFrame(data, columns=['seq_len', 'image_path', 'pose'])
    # Disable shuffle and sort to preserve original order
    if shuffle:
        df = df.sample(frac=1)
    if sort:
        df = df.sort_values(by=['seq_len'], ascending=False)
    return df
    

def get_partition_data_info(partition, folder_list, seq_len_range, overlap, sample_times=1, pad_y=False, shuffle=False, sort=True, config=None):
    """
    Generate partitioned data info dataframes.
    
    Args:
        partition (float): Partition ratio for validation set.
        folder_list (list): List of video folders to process.
        seq_len_range (tuple): Range of sequence lengths (min, max).
        overlap (int): Overlap between sequences.
        sample_times (int): Number of times to sample sequences.
        pad_y (bool): Whether to pad ground truth with zeros.
        shuffle (bool): Whether to shuffle the dataframe.
        sort (bool): Whether to sort by sequence length.
        config (dict): Configuration dictionary from config.yaml.
    """
    X_path = [[], []]
    Y = [[], []]
    X_len = [[], []]
    df_list = []

    for part in range(2):
        for folder in folder_list:
            start_t = time.time()
            poses = np.load(os.path.join(config["deepvo"]["pose_dir"], f"{folder}.npy"))  # (n_images, 6)
            fpaths = glob.glob(os.path.join(config["deepvo"]["image_dir"], f"{folder}/image_02/*.png"))
            fpaths.sort()

            n_val = int((1-partition)*len(fpaths))
            st_val = int((len(fpaths)-n_val)/2)
            ed_val = st_val + n_val
            print('st_val: {}, ed_val:{}'.format(st_val, ed_val))
            if part == 1:
                fpaths = fpaths[st_val:ed_val]
                poses = poses[st_val:ed_val]
            else:
                fpaths = fpaths[:st_val] + fpaths[ed_val:]
                poses = np.concatenate((poses[:st_val], poses[ed_val:]), axis=0)

            assert(overlap < min(seq_len_range))
            n_frames = len(fpaths)
            min_len, max_len = seq_len_range[0], seq_len_range[1]
            for i in range(sample_times):
                start = 0
                while True:
                    n = np.random.random_integers(min_len, max_len)
                    if start + n < n_frames:
                        x_seg = fpaths[start:start+n] 
                        X_path[part].append(x_seg)
                        if not pad_y:
                            Y[part].append(poses[start:start+n])
                        else:
                            pad_zero = np.zeros((max_len-n, 6))
                            padded = np.concatenate((poses[start:start+n], pad_zero))
                            Y[part].append(padded.tolist())
                    else:
                        print('Last %d frames is not used' %(start+n-n_frames))
                        break
                    start += n - overlap
                    X_len[part].append(len(x_seg))
            print('Folder {} finish in {} sec'.format(folder, time.time()-start_t))
        
        data = {'seq_len': X_len[part], 'image_path': X_path[part], 'pose': Y[part]}
        df = pd.DataFrame(data, columns=['seq_len', 'image_path', 'pose'])
        if shuffle:
            df = df.sample(frac=1)
        if sort:
            df = df.sort_values(by=['seq_len'], ascending=False)
        df_list.append(df)
    return df_list

class SortedRandomBatchSampler(Sampler):
    def __init__(self, info_dataframe, batch_size, drop_last=False):
        self.df = info_dataframe
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.unique_seq_lens = sorted(self.df.iloc[:].seq_len.unique(), reverse=True)
        self.len = 0
        for v in self.unique_seq_lens:
            n_sample = len(self.df.loc[self.df.seq_len == v])
            n_batch = int(n_sample / self.batch_size)
            if not self.drop_last and n_sample % self.batch_size != 0:
                n_batch += 1
            self.len += n_batch

    def __iter__(self):
        list_batch_indexes = []
        start_idx = 0
        for v in self.unique_seq_lens:
            n_sample = len(self.df.loc[self.df.seq_len == v])
            n_batch = int(n_sample / self.batch_size)
            if not self.drop_last and n_sample % self.batch_size != 0:
                n_batch += 1
            rand_idxs = (start_idx + torch.randperm(n_sample)).tolist()
            tmp = [rand_idxs[s*self.batch_size: s*self.batch_size+self.batch_size] for s in range(0, n_batch)]
            list_batch_indexes += tmp
            start_idx += n_sample
        return iter(list_batch_indexes)

    def __len__(self):
        return self.len

class ImageSequenceDataset(Dataset):
    def __init__(self, info_dataframe, resize_mode='crop', new_size=None, img_mean=None, img_std=(1,1,1), minus_point_5=False, config=None):
        """
        Initialize the ImageSequenceDataset.
        
        Args:
            info_dataframe (pd.DataFrame): Dataframe containing sequence info.
            resize_mode (str): Mode for resizing images ('crop' or 'rescale').
            new_size (tuple): Target image size (height, width).
            img_mean (tuple): Mean values for normalization.
            img_std (tuple): Standard deviation values for normalization.
            minus_point_5 (bool): Whether to subtract 0.5 from image values.
            config (dict): Configuration dictionary from config.yaml.
        """
        transform_ops = []
        if resize_mode == 'crop':
            transform_ops.append(transforms.CenterCrop((new_size[0], new_size[1])))
        elif resize_mode == 'rescale':
            transform_ops.append(transforms.Resize((new_size[0], new_size[1])))
        transform_ops.append(transforms.ToTensor())
        self.transformer = transforms.Compose(transform_ops)
        self.minus_point_5 = minus_point_5
        self.normalizer = transforms.Normalize(mean=img_mean, std=img_std) if img_mean else None
        
        self.data_info = info_dataframe
        self.seq_len_list = list(self.data_info.seq_len)
        self.image_arr = np.asarray(self.data_info.image_path)
        self.groundtruth_arr = np.asarray(self.data_info.pose)

    def __getitem__(self, index):
        raw_groundtruth = np.hsplit(self.groundtruth_arr[index], np.array([6]))	
        groundtruth_sequence = raw_groundtruth[0]
        groundtruth_rotation = raw_groundtruth[1][0].reshape((3, 3)).T

        groundtruth_sequence = torch.FloatTensor(groundtruth_sequence)
        groundtruth_sequence[1:] = groundtruth_sequence[1:] - groundtruth_sequence[0]

        for gt_seq in groundtruth_sequence[1:]:
            location = torch.FloatTensor(groundtruth_rotation.dot(gt_seq[3:].numpy()))
            gt_seq[3:] = location[:]

        groundtruth_sequence[2:] = groundtruth_sequence[2:] - groundtruth_sequence[1:-1]

        for gt_seq in groundtruth_sequence[1:]:
            gt_seq[0] = normalize_angle_delta(gt_seq[0])
			
        image_path_sequence = self.image_arr[index]
        sequence_len = torch.tensor(self.seq_len_list[index])
        #print(f"Loading images from {image_path_sequence[0]}")  # Debug print
        
        image_sequence = []
        for img_path in image_path_sequence:
            if not os.path.exists(img_path):
                print(f"Image not found: {img_path}")
                continue
            img_as_img = Image.open(img_path)
            img_as_tensor = self.transformer(img_as_img)
            if self.minus_point_5:
                img_as_tensor = img_as_tensor - 0.5
            if self.normalizer:
                img_as_tensor = self.normalizer(img_as_tensor)
            img_as_tensor = img_as_tensor.unsqueeze(0)
            image_sequence.append(img_as_tensor)
        image_sequence = torch.cat(image_sequence, 0)
        return (sequence_len, image_sequence, groundtruth_sequence)

    def __len__(self):
        return len(self.data_info.index)

def normalize_angle_delta(angle):
    if(angle > np.pi):
        angle = angle - 2 * np.pi
    elif(angle < -np.pi):
        angle = 2 * np.pi + angle
    return angle

# Example of usage (optional, can be removed if not needed)
if __name__ == '__main__':
    start_t = time.time()
    overlap = 1
    sample_times = 1
    folder_list = ['00']
    seq_len_range = [5, 7]
    df = get_data_info(folder_list, seq_len_range, overlap, sample_times, config={'deepvo': {'image_dir': '/path/to/images/', 'pose_dir': '/path/to/poses/'}})
    print('Elapsed Time (get_data_info): {} sec'.format(time.time()-start_t))
    n_workers = 4
    resize_mode = 'crop'
    new_size = (150, 600)
    img_mean = (-0.14968217427134656, -0.12941663107068363, -0.1320610301921484)
    dataset = ImageSequenceDataset(df, resize_mode, new_size, img_mean, config={'deepvo': {'image_dir': '/path/to/images/', 'pose_dir': '/path/to/poses/'}})
    sorted_sampler = SortedRandomBatchSampler(df, batch_size=4, drop_last=True)
    dataloader = DataLoader(dataset, batch_sampler=sorted_sampler, num_workers=n_workers)
    print('Elapsed Time (dataloader): {} sec'.format(time.time()-start_t))

    for batch in dataloader:
        s, x, y = batch
        print('='*50)
        print('len:{}\nx:{}\ny:{}'.format(s, x.shape, y.shape))
    
    print('Elapsed Time: {} sec'.format(time.time()-start_t))
    print('Number of workers = ', n_workers)