import json
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from vocabulary import Vocabulary


class ImageTextDataset(Dataset):
    def __init__(self, root_dir, label_path, vocabulary: Vocabulary, max_seq_len, transform=lambda x: x, max_cache_memory=16 * 1024 ** 3):
        """
        Args:
            root_dir (string): Root directory of the dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        with open(label_path, 'rb') as fp:
            self.filenames, self.labels = zip(*json.load(fp).items())

        self.root_dir = root_dir
        self.max_seq_len = max_seq_len

        self.vocabulary = vocabulary
        self.transform = transform

        self.image_cache = {}
        image_path = os.path.join(self.root_dir, 'images', self.filenames[0])
        image = self.transform(Image.open(image_path))
        image_size = image.element_size() * image.nelement()  # 每一张图像占用内存大小（所有图像transform后形状一致）
        self.max_cache_size = max_cache_memory // image_size

    def __len__(self):
        return len(self.filenames)

    def sample(self):
        idx = random.choice(range(len(self)))
        return self.get_pair(idx)

    def get_pair(self, idx):
        image_path = os.path.join(self.root_dir, 'images', self.filenames[idx])
        text_label = self.labels[idx]
        return image_path, text_label

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = os.path.join(self.root_dir, 'images', self.filenames[idx])
        if image_path not in self.image_cache:
            image = self.transform(Image.open(image_path))
            if len(self.image_cache) < self.max_cache_size:  # 60GB
                self.image_cache[image_path] = image  # 缓存图像
        else:
            image = self.image_cache[image_path]  # 从缓存中获取图像

        indices = self.vocabulary.encode(self.labels[idx])
        actual_length = len(indices)
        indices = self.vocabulary.pad_sequence(indices, self.max_seq_len)
        seq = torch.LongTensor(np.array(indices))
        return image, seq, actual_length
