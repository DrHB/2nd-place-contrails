import numpy as np
import cv2, os, gc
import torch

from torch.utils.data import Dataset, DataLoader
import tifffile
import albumentations as A


def rle_encode_less_memory(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    This simplified method requires first and last pixel to be zero
    """
    pixels = img.T.flatten()

    # This simplified method requires first and last pixel to be zero
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


def read_record(path, files=["band_11", "band_14", "band_15", "human_pixel_masks"]):
    record_data = {}
    for x in files:
        record_data[x] = np.load(os.path.join(path, f"{x}.npy"))

    return record_data


def normalize_range(data, bounds):
    """Maps data to the range [0, 1]."""
    return (data - bounds[0]) / (bounds[1] - bounds[0])


def get_false_color(record_data, N_TIMES_BEFORE=4, full=False):
    _T11_BOUNDS = (243, 303)
    _CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
    _TDIFF_BOUNDS = (-4, 2)

    r = normalize_range(record_data["band_15"] - record_data["band_14"], _TDIFF_BOUNDS)
    g = normalize_range(
        record_data["band_14"] - record_data["band_11"], _CLOUD_TOP_TDIFF_BOUNDS
    )
    b = normalize_range(record_data["band_14"], _T11_BOUNDS)
    false_color = np.clip(np.stack([r, g, b], axis=2), 0, 1)
    if full:
        return false_color
    else:
        return false_color[..., N_TIMES_BEFORE]


class ContrailsDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.fnames = sorted([fname for fname in os.listdir(self.path)])

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        idx = idx % len(self.fnames)
        record = read_record(
            os.path.join(self.path, self.fnames[idx]), ["band_11", "band_14", "band_15"]
        )
        img = get_false_color(record, full=True)
        h, w, c, t = img.shape  # 256,256,3,8
        img = img.reshape(h, w, t * c)

        # img = img.reshape(*img.shape[:2],self.nc,-1)#[:,:,:,:6]
        # img = img.reshape(*img.shape[:2],-1)
        img = (img.clip(0, 1) * 255).astype(np.uint8).astype(np.float32) / 255

        # img = cv2.resize(img, (2*img.shape[1],2*img.shape[0]), interpolation=cv2.INTER_CUBIC)
        img = img2tensor(img)
        img = img.view(c, -1, *img.shape[1:])

        return img, self.fnames[idx]


def get_aug():
    return A.Compose(
        [
            A.ShiftScaleRotate(
                shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.75
            ),
            A.OneOf(
                [
                    A.RandomGamma(gamma_limit=(50, 150), always_apply=True),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.1, contrast_limit=0.2, always_apply=True
                    ),
                ],
                p=0.5,
            ),
            A.OneOf(
                [
                    A.MotionBlur(always_apply=True),
                    A.GaussianBlur(always_apply=True),
                ],
                p=0.25,
            ),
        ],
        p=1,
    )


class ContrailsDatasetV0(Dataset):
    def __init__(self, path, train=True, tfms=None, repeat=1):
        self.path = os.path.join(path, "train_adj2" if train else "val_adj2")
        self.fnames = sorted(
            [
                fname
                for fname in os.listdir(self.path)
                if fname.split(".")[0].split("_")[-1] == "img"
            ]
        )
        self.train, self.tfms = train, tfms
        self.nc = 3
        self.repeat = repeat

    def __len__(self):
        return self.repeat * len(self.fnames)

    def __getitem__(self, idx):
        idx = idx % len(self.fnames)
        img = tifffile.imread(os.path.join(self.path, self.fnames[idx]))
        img = img.reshape(*img.shape[:2], self.nc, -1)[:, :, :, :5]
        img = img.reshape(*img.shape[:2], -1)
        mask = tifffile.imread(
            os.path.join(self.path, self.fnames[idx].replace("img", "mask"))
        )

        if self.tfms is not None:
            augmented = self.tfms(image=img, mask=mask)
            img, mask = augmented["image"], augmented["mask"]

        img = cv2.resize(
            img, (1 * img.shape[1], 1 * img.shape[0]), interpolation=cv2.INTER_CUBIC
        )
        img, mask = img2tensor(img / 255), img2tensor(mask / 255)
        img = img.view(self.nc, -1, *img.shape[1:])

        return img, mask
