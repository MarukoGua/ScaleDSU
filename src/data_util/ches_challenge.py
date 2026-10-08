import logging

import numpy as np
import h5py
from torch.utils.data import Dataset

from .preprocess import notch_many_fft

# KEYS TO GET DATA FROM H5PY FORMAT FILE
PROFILING_TRACES = 'Profiling_traces/traces'
PROFILING_META = 'Profiling_traces/metadata'

ATTACK_TRACES = "Attack_traces/traces"
ATTACK_META = 'Attack_traces/metadata'

EVALUATION_TRACES = lambda x: f"Evaluation_traces_{x}/traces"
EVALUATION_META = lambda x: f"Evaluation_traces_{x}/metadata"



AES_Sbox = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
])

class CHESChallengeDataset(Dataset):
    def __init__(self, trace, label, meta_info, target_byte=0, dataset_name='ches_challenge'):
        self.trace = trace
        self.label = label
        self.meta = meta_info
        self.target_byte = target_byte
        self.name = dataset_name
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        norm_trace = self.trace[idx, :].astype(float)
        label = self.label[idx]

        sample = {"norm_trace": norm_trace, "label": label}
        return sample

    def apply_preprocess(self, process_fun, *args):
        if self.trace.shape[0] > 10_0000:
            n_sub = 10
            t_sub = int(self.trace.shape[0] / 10)
            for i in range(n_sub):
                start = i * t_sub
                end = min((i+1)*t_sub, self.trace.shape[0])
                y = notch_many_fft(
                    self.trace[start:end, ...], 
                    fs=1000, 
                    freqs=[10, 20, 25, 30, 40, 43.15, 60, 80, 125],
                    Q=100,
                    taper_ratio=0.5,
                    atten_db=20,
                    mix=1.0
                    )

                self.trace[i*t_sub:(i+1)*t_sub] = process_fun(y, *args)
        else:
            y = notch_many_fft(
                self.trace, 
                fs=1000, 
                freqs=[10, 20, 25, 30, 40, 43.15, 60, 80, 125],
                Q=100,
                taper_ratio=0.5,
                atten_db=20,
                mix=1.0
                )
            self.trace = process_fun(y, *args)

    def generate_labels_metric(self, indices, n_hyp):
        plaintexts = np.tile(self.meta['plaintext'][:, self.target_byte][indices, None], n_hyp)
        hyp_keys= np.tile(np.arange(0, n_hyp)[:, None], indices.shape).T
        labels = AES_Sbox[hyp_keys ^ plaintexts]
        return labels

def load_ches_challenge_from_file(file, keys, n_train=None, n_test=None):
    in_file = h5py.File(file, "r")
    n_total = len(in_file[keys[1]])
    n_needed = (n_train if n_train is not None else 0) + (n_test if n_test is not None else 0)

    # Read only the requested number of traces (a random subset) instead of the
    # whole dataset, to keep peak memory low (e.g. ~110k instead of 500k traces).
    if n_needed <= 0 or n_needed >= n_total:
        idx = np.arange(n_total)
    else:
        idx = np.sort(np.random.permutation(n_total)[:n_needed])

    def _read(dataset, idx):
        if len(idx) >= n_total:
            return np.array(dataset)
        if len(idx) <= 10_000:
            # few rows: direct fancy indexing is fast
            return np.array(dataset[idx])
        # many rows: per-row h5py fancy indexing is very slow on contiguous
        # datasets, so pull rows via fast contiguous block reads instead.
        block = 100_000
        n_blocks = (n_total + block - 1) // block
        parts = []
        for b in range(n_blocks):
            start = b * block
            end = min((b + 1) * block, n_total)
            sel = idx[(idx >= start) & (idx < end)] - start
            if len(sel) == 0:
                continue
            parts.append(np.array(dataset[start:end])[sel])
        return np.concatenate(parts, axis=0)

    X = _read(in_file[keys[0]], idx)
    metadata = _read(in_file[keys[1]], idx)
    Y = metadata['labels']
    in_file.close()

    train, test = None, None
    offset = 0

    if n_train is not None:
        train = (X[offset:offset + n_train], Y[offset:offset + n_train], metadata[offset:offset + n_train])
        offset += n_train

    if n_test is not None:
        test = (X[offset:offset + n_test], Y[offset:offset + n_test], metadata[offset:offset + n_test])

    return train, test
    

def load_ches_challenge(args, mode, n_train=100_000, n_test=100_000, is_cdpa=False, tgts=None):
    file = args.data_file
    src_file = f'{file}/CHES_Challenge.h5'
    tgt_file = f'{file}/CHES_Challenge_evaluation.h5'

    if mode == 'train':
        keys = [PROFILING_TRACES, PROFILING_META]
        src_train_data, src_test_data = load_ches_challenge_from_file(
            src_file, 
            keys,
            n_train=n_train,
            n_test=n_test
        )
        src_train_data = CHESChallengeDataset(*src_train_data, target_byte=args.target_byte)
        src_test_data = CHESChallengeDataset(*src_test_data, target_byte=args.target_byte)  

        return src_train_data, src_test_data
        

    elif mode == 'test':
        tgt_datasets = dict()
        tgts = args.targets.split(',') if tgts is None else tgts
        if '0' in tgts:
            keys = [ATTACK_TRACES, ATTACK_META]
            _, tgt_data = load_ches_challenge_from_file(src_file, keys, n_test=n_test)
            if is_cdpa:
                n=100
                X, Y, meta = tgt_data
                val_tune = CHESChallengeDataset(X[:n], Y[:n], meta[:n], target_byte=args.target_byte)
                val_test = CHESChallengeDataset(X[n:], Y[n:], meta[n:], target_byte=args.target_byte)
                tgt_datasets['tgt-0'] = (val_tune, val_test)
            else:
                tgt_datasets['tgt-0'] = CHESChallengeDataset(*tgt_data, target_byte=args.target_byte)
            # tgts.remove('0')

        for tgt in tgts:
            if tgt == '0':
                continue
            keys = [EVALUATION_TRACES(tgt), EVALUATION_META(tgt)]
            _, tgt_data = load_ches_challenge_from_file(tgt_file, keys, n_test=n_test)
            if is_cdpa:
                n=100
                X, Y, meta = tgt_data
                val_tune = CHESChallengeDataset(X[:n], Y[:n], meta[:n], target_byte=args.target_byte)
                val_test = CHESChallengeDataset(X[n:], Y[n:], meta[n:], target_byte=args.target_byte)
                tgt_datasets[f'tgt-{tgt}'] = (val_tune, val_test)
            else:
                tgt_datasets[f'tgt-{tgt}'] = CHESChallengeDataset(*tgt_data, target_byte=args.target_byte)
        
        return tgt_datasets
    else:
        print('wrong mode!!! exit!!!')
        exit(1)
