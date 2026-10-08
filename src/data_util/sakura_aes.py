import os
import logging

import numpy as np
import torch

from torch.utils.data import Dataset
import data_util

PROFILING_TRACES = "X_train.npy"
PROFILING_LABELS = "Y_train.npy"

ATTACK_TRACES = "X_attack.npy"
ATTACK_LABELS = "Y_attack.npy"
ATTACK_CIPHER = "ciphertexts_attack.npy"


InvSbox = np.array([82, 9, 106, 213, 48, 54, 165, 56, 191, 64, 163, 158, 129, 243, 215, 251, 124, 227, 57, 130, 155, 47, 255, 135,
           52, 142, 67, 68, 196, 222, 233, 203, 84, 123, 148, 50, 166, 194, 35, 61,238, 76, 149, 11, 66, 250, 195, 78, 8,
           46, 161, 102, 40, 217, 36, 178, 118, 91, 162, 73, 109, 139, 209, 37, 114, 248, 246, 100, 134, 104, 152, 22, 212,
           164, 92, 204, 93, 101, 182, 146, 108, 112, 72, 80, 253, 237, 185, 218, 94, 21, 70, 87, 167, 141, 157, 132, 144,
           216, 171, 0, 140, 188, 211, 10, 247, 228, 88, 5, 184, 179, 69, 6, 208, 44, 30, 143, 202, 63, 15, 2, 193, 175, 189,
           3, 1, 19, 138, 107, 58, 145, 17, 65, 79, 103, 220, 234, 151, 242, 207, 206, 240, 180, 230, 115, 150, 172, 116, 34,
           231, 173, 53, 133, 226, 249, 55, 232, 28, 117, 223, 110, 71, 241, 26, 113, 29, 41, 197, 137, 111, 183, 98, 14, 170,
           24,190, 27, 252, 86, 62, 75, 198, 210, 121, 32, 154, 219, 192, 254, 120, 205, 90, 244, 31, 221, 168, 51, 136, 7,
           199, 49, 177, 18, 16, 89, 39, 128, 236, 95, 96, 81, 127, 169, 25, 181,74, 13, 45, 229, 122, 159, 147, 201, 156,
           239, 160, 224, 59, 77, 174, 42, 245, 176, 200, 235, 187, 60, 131, 83, 153, 97, 23, 43, 4, 126, 186, 119, 214, 38,
           225, 105, 20, 99, 85, 33,12, 125])


def get_real_key(device):
    keys = [0x21, 0xCD, 0x8F]
    device_id = int(device[-1])
    return keys[device_id-1]
        

class SakuraAESDataset(Dataset):
    def __init__(self, traces, labels, ciphertext, real_key, dataset_name='sakura_aes', **kwargs):
        self.origin_trace = traces
        self.label = labels
        self.real_key = real_key
        self.normalized_trace = None
        if ciphertext is None:
            self.ciphertext = np.zeros(shape=(len(labels), 16))
        else:
            self.ciphertext = ciphertext
        self.target_byte = 0
        self.name = dataset_name

        self.get_metadata()

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        norm_trace = self.origin_trace[idx, :].astype(float)

        if self.normalized_trace is not None:
            norm_trace = self.normalized_trace[idx, :].astype(float)

        label = self.label[idx]
            
        sample = {
            "norm_trace": norm_trace, 
            "label": label
            }
        return sample
    
    def get_metadata(self):
        keys = np.array((self.real_key, ) * len(self.label))
        metadata_type = np.dtype([
            ('ciphertext', self.ciphertext.dtype, (16, )),
            ('key', keys.dtype, (1, )),
        ])
        self.meta = np.array(list(zip(self.ciphertext, keys)), dtype=metadata_type)
    

    def apply_preprocess(self, process_fun, *args):
        self.normalized_trace = process_fun(self.origin_trace, *args)

    def generate_labels_metric(self, indices, n_hyp):
        ciphertext_1 = np.tile(self.meta['ciphertext'][:, 1][indices, None], n_hyp)
        ciphertext_5 = np.tile(self.meta['ciphertext'][:, 5][indices, None], n_hyp)
        hyp_keys = np.tile(np.arange(0, n_hyp)[:, None], indices.shape).T
        labels = InvSbox[hyp_keys ^ ciphertext_1] ^  ciphertext_5
        return labels
    

def load_sakura_from_file(in_dir, files):
    X = np.load(os.path.join(in_dir, files[0]))
    Y = np.load(os.path.join(in_dir, files[1]))
    if len(files) == 3:
        ciphertext = np.load(os.path.join(in_dir, files[2])).astype(np.int64)
    else:
        ciphertext=None

    return (X, Y, ciphertext)


def load_sakura_aes(args, mode):
    
    if mode == 'train':
        data_dir = args.data_file
        source_file = f'{data_dir}/device{args.source}'
        source_real_key = get_real_key(args.source)
        keys = [PROFILING_TRACES, PROFILING_LABELS]
        src_train_data = load_sakura_from_file(source_file, keys)
        src_train_data = SakuraAESDataset(*src_train_data, real_key=source_real_key)

        keys = [ATTACK_TRACES, ATTACK_LABELS, ATTACK_CIPHER]
        src_val_data = load_sakura_from_file(source_file, keys)
        src_val_data = SakuraAESDataset(*src_val_data, real_key=source_real_key)
        return src_train_data, src_val_data
    
    elif mode == 'test':
        data_dir = args.data_file
        targets = args.targets.split(',')
        test_datasets = {}
        for target in targets:
            file = f'{data_dir}/device{target}'
            real_key = get_real_key(target)
            keys = [ATTACK_TRACES, ATTACK_LABELS, ATTACK_CIPHER]
        
            test_data = load_sakura_from_file(file, keys)
            test_data = SakuraAESDataset(*test_data, real_key=real_key)
            test_datasets[target] = test_data
        return test_datasets