import numpy as np
import os
import tifffile
from tqdm import tqdm
from pathlib import Path
import argparse
import json

def read_config_file(file_path):
    with open(file_path, "r") as f:
        config_data = json.load(f)
    return config_data


def read_record(path, files=["band_11", "band_14", "band_15", "human_pixel_masks"]):
    record_data = {}
    for x in files:
        record_data[x] = np.load(os.path.join(path,f"{x}.npy"))

    return record_data


def normalize_range(data, bounds):
    """Maps data to the range [0, 1]."""
    return (data - bounds[0]) / (bounds[1] - bounds[0])


def get_false_color(record_data, N_TIMES_BEFORE = 4, full=False):
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
    
    
def multiframe(PATH, OUT, valid=False):
    for fname in tqdm(os.listdir(PATH)):
        record = read_record(os.path.join(PATH,fname),
                            ["band_11", "band_14", "band_15", "human_pixel_masks"])
        img = get_false_color(record, full=True)
        if valid:
            mask = record['human_pixel_masks']
        else:
            mask = record['human_pixel_masks'].mean(-1)
        
        h,w,c,t = img.shape #256,256,3,8
        img = img.reshape(h,w,t*c)
        img_adj = (img*255).clip(0,255).astype(np.uint8)
        tifffile.imwrite(os.path.join(OUT,fname + '_img.tif'), img_adj)
        mask = (255*mask).astype(np.uint8)
        tifffile.imwrite(os.path.join(OUT,fname + '_mask.tif'), mask)
        
def singleframe(PATH, OUT, valid = False):
    for fname in tqdm(os.listdir(PATH)):
        record = read_record(os.path.join(PATH,fname),
                            ["band_11", "band_14", "band_15", "human_pixel_masks"])
        img = get_false_color(record, full=False)
        if valid:
            mask = record['human_pixel_masks']
        else:
            mask = record['human_pixel_masks'].mean(-1)
        
        #h,w,c,t = img.shape #256,256,3,8
        #img = img.reshape(h,w,t*c)
        img_adj = (img*255).clip(0,255).astype(np.uint8)
        tifffile.imwrite(os.path.join(OUT,fname + '_img.tif'), img_adj)
        mask = (255*mask).astype(np.uint8)
        tifffile.imwrite(os.path.join(OUT,fname + '_mask.tif'), mask)
        
        
def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Dataset."
    )
    parser.add_argument("config_file", type=str, help="Path to the JSON config file.")
    args = parser.parse_args()
    config_file_path = args.config_file
    config_data = read_config_file(config_file_path)

    OUT = Path(config_data["PATH"])/'train_adj2'
    os.makedirs(OUT, exist_ok=True)
    multiframe(Path(config_data["PATH"])/'train', OUT, valid=False)

    OUT = Path(config_data["PATH"])/'val_adj2'
    os.makedirs(OUT, exist_ok=True)
    multiframe(Path(config_data["PATH"])/'validation', OUT, valid=True)


    OUT = Path(config_data["PATH"])/'train_adj2single'
    os.makedirs(OUT, exist_ok=True)
    singleframe(Path(config_data["PATH"])/'train', OUT, valid=False)

    OUT = Path(config_data["PATH"])/'val_adj2single'
    os.makedirs(OUT, exist_ok=True)
    singleframe(Path(config_data["PATH"])/'validation', OUT, valid=True)
    
if __name__ == "__main__":
    main()