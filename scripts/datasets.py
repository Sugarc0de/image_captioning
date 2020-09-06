import h5py
import numpy as np
import os
import json
from torch.utils.data import Dataset
from torchvision import transforms, utils


class CaptionsDataset(Dataset):
    """Captions dataset"""

    def __init__(self, data_path, split, transform=None):
        assert split in ["TRAIN", "VAL", "TEST"]

        CAPTIONS_PER_IMAGE = 5
        MIN_WORD_FREQ = 5
        self._mean = [0.485, 0.456, 0.406]
        self._std = [0.229, 0.224, 0.225]

        with open(
            os.path.join(
                data_path,
                f"{split}_CAPTIONS_flickr30k_{CAPTIONS_PER_IMAGE}_cap_per_img_{MIN_WORD_FREQ}_min_word_freq.json",
            ),
            "r",
        ) as f:
            self.enc_captions = json.load(f)

        with open(
            os.path.join(
                data_path,
                f"{split}_CAPLENS_flickr30k_{CAPTIONS_PER_IMAGE}_cap_per_img_{MIN_WORD_FREQ}_min_word_freq.json",
            ),
            "r",
        ) as f:
            self.caplens = json.load(f)

        self.h = h5py.File(
            os.path.join(
                data_path,
                f"{split}_IMAGES_flickr30k_{CAPTIONS_PER_IMAGE}_cap_per_img_{MIN_WORD_FREQ}_min_word_freq.hdf5",
            ),
            "r",
        )
        self.images = np.array(self.h["images"]).astype("uint8")

        self.transform = transform

    def __len__(self):
        return len(self.enc_captions[0]) * len(self.enc_captions[0])

    def __getitem__(self, idx):
        """For the idx (th) caption, it is the (idx / CAPTIONS_PER_IMAGE) image"""

        caption = self.enc_captions[idx]
        caplen = self.caplens[idx]

        image = self.images[idx, :, :, :]
        assert np.max(image) <= 255 and np.min(image) >= 0

        image = image // 255.0

        pixels = (image - self._mean) / self._std

        return pixels, caption, caplen
