import h5py
import math
import torch
import numpy as np
import os
import json
from torch.utils.data import Dataset
from torchvision import transforms, utils


class CaptionsDataset(Dataset):
    """Captions dataset"""

    def __init__(self, data_path, split, transform=None):
        assert split in ["TRAIN", "VAL", "TEST"]

        self.CAPTIONS_PER_IMAGE = 5
        MIN_WORD_FREQ = 5

        with open(
            os.path.join(
                data_path,
                f"{split}_CAPTIONS_flickr30k_{self.CAPTIONS_PER_IMAGE}_cap_per_img_{MIN_WORD_FREQ}_min_word_freq.json",
            ),
            "r",
        ) as f:
            self.enc_captions = json.load(f)

        with open(
            os.path.join(
                data_path,
                f"{split}_CAPLENS_flickr30k_{self.CAPTIONS_PER_IMAGE}_cap_per_img_{MIN_WORD_FREQ}_min_word_freq.json",
            ),
            "r",
        ) as f:
            self.caplens = json.load(f)

        self.h = h5py.File(
            os.path.join(
                data_path,
                f"{split}_IMAGES_flickr30k_{self.CAPTIONS_PER_IMAGE}_cap_per_img_{MIN_WORD_FREQ}_min_word_freq.hdf5",
            ),
            "r",
        )

        self.transform = transform

    def __len__(self):
        return len(self.enc_captions[0]) * len(self.enc_captions[0])

    def __getitem__(self, idx):
        """For the idx (th) caption, it is the (idx / CAPTIONS_PER_IMAGE) image"""

        caption = self.enc_captions[idx]
        caplen = self.caplens[idx]
        print(f"The {np.int(np.floor(idx/self.CAPTIONS_PER_IMAGE))} th image...")
        assert self.h.attrs["captions_per_image"] == self.CAPTIONS_PER_IMAGE
        image = np.array(
            self.h["/home/ec2-user/SageMaker/efs/200005/images"][
                int(idx // self.CAPTIONS_PER_IMAGE)
            ]
        )

        assert np.max(image) <= 255 and np.min(image) >= 0

        image = np.clip(image / 255.0, 0, 1)

        image = np.transpose(image, (1, 2, 0))

        if self.transform:
            image = self.transform(image)

        return image, caption, caplen
