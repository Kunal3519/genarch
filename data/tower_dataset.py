"""Tower datasets for the GenArch pipeline (myarch only).

    MotionDataset       sliding-window coord snippets -- VQ-VAE tokenizer training
    Image2TowerDataset  4-view DINOv2 patch features + tower -- masked transformer
"""
import os
from os.path import join as pjoin
import random

import numpy as np
from torch.utils import data
from tqdm import tqdm


class MotionDataset(data.Dataset):
    """Sliding-window snippet dataset over per-tower coord sequences.

    Used by the RVQ tokenizer trainer. Returns z-normalised motion windows of
    shape (window_size, dim_pose).
    """

    def __init__(self, opt, mean, std, split_file):
        self.opt = opt

        self.data = []
        self.lengths = []
        id_list = []
        with open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if motion.shape[0] < opt.window_size:
                    continue
                self.lengths.append(motion.shape[0] - opt.window_size)
                self.data.append(motion)
            except Exception as e:
                print(e)

        self.cumsum = np.cumsum([0] + self.lengths)

        if opt.is_train:
            np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
            np.save(pjoin(opt.meta_dir, 'std.npy'), std)

        self.mean = mean
        self.std = std
        print('Total number of motions {}, snippets {}'.format(len(self.data), self.cumsum[-1]))

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
        motion = self.data[motion_id][idx:idx + self.opt.window_size]
        motion = (motion - self.mean) / self.std
        return motion



class Image2TowerDataset(data.Dataset):
    """Dataset for image-conditioned tower generation.

    Returns (img_feat, motion [max_motion_length, dim_pose], m_length) per sample.
    img_feat is a pre-encoded feature loaded from opt.image_feat_dir
    (e.g. CLIP 768-d, DINOv2 CLS 4096-d, or DINOv2 patch (4,256,1024)).
    """

    def __init__(self, opt, mean, std, split_file):
        self.opt = opt
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 1

        data_dict = {}
        id_list = []
        with open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        name_list, length_list = [], []
        skipped = 0
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if len(motion) < min_motion_len or len(motion) > self.max_motion_length:
                    continue
                img_feat_path = pjoin(opt.image_feat_dir, name + '.npy')
                if not os.path.exists(img_feat_path):
                    skipped += 1
                    continue
                # stream_img_feat: don't preload the (4,256,1024) feature (~4MB/tower)
                # into RAM (160k towers -> ~640GB/process, prohibits parallel runs);
                # load it per-item in __getitem__ instead. Default keeps preloading.
                if getattr(opt, 'stream_img_feat', False):
                    data_dict[name] = {'motion': motion, 'length': len(motion), 'img_feat': None}
                else:
                    img_feat = np.load(img_feat_path).astype(np.float32)
                    data_dict[name] = {'motion': motion, 'length': len(motion), 'img_feat': img_feat}
                name_list.append(name)
                length_list.append(len(motion))
            except Exception:
                pass

        if skipped:
            print(f'Skipped {skipped} samples (missing image features)')

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.pointer = 0
        print(f'Image2TowerDataset: {len(self.data_dict)} towers loaded')

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        name = self.name_list[idx]
        data = self.data_dict[name]
        motion, m_length, img_feat = data['motion'], data['length'], data['img_feat']
        if img_feat is None:   # stream_img_feat mode: load per-item
            img_feat = np.load(pjoin(self.opt.image_feat_dir, name + '.npy')).astype(np.float32)

        if getattr(self.opt, 'deterministic_crop', False):
            # Towers are a fixed generator ordering, not time-shiftable motion:
            # random crops jitter the sequence boundaries every epoch and train a
            # fuzzy ending (40% of vote-FN land in the last sequence decile).
            # Always keep the full sequence from index 0.
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
            m_length = max(m_length, self.opt.unit_length)
            idx = 0
        else:
            if self.opt.unit_length < 10:
                coin = np.random.choice(['single', 'single', 'double'])
            else:
                coin = 'single'

            if coin == 'double':
                m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
            else:
                m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
            m_length = max(m_length, self.opt.unit_length)

            idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate(
                [motion, np.zeros((self.max_motion_length - m_length, motion.shape[1]))], axis=0)

        return img_feat, motion, m_length

