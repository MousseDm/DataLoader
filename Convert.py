import os
import numpy as np 
import cv2
from torch.utils.data import Dataset, DataLoader
from pathlib import Path 

class Convert(Dataset):
    def __init__(self, source_img_folder, source_mask_folder, target_folder, resolution = 1024):
        self.source_img_folder = Path(source_img_folder)
        self.source_out_folder = Path(source_out_folder) / "segmentation" 
        self.target_folder = Path(target_folder)
        self.resolution = resolution

        self.image_folders = sorted([f for f in self.source_img_folder.iterdir() if f.is_dir()])

        self.annotations_folder = self.target_folder / Annotations
        self.img_target_folder = self.target_folder / JPEGImages

    def __len__(self):
        return len(self.image_folders)
    
    def __getitem__(self, idx):
        img_folder = self.image_folders[idx] #例如fold2

        img_paths = sorted(img_folder.glob("*.jpg")) #fold2中每个jpg的路径的合集
        
        images = []

        for img_path in img_paths:
            img_id = img_path.stem #编码
            mask_path = self.source_out_folder / img_id / "mask.npy"

            if not mask_path.exists():
                continue

            img = cv2.imread(str(img_path))
            img_resized = cv2.resize(img, (self.resolution, self.resolution))

            mask = np.load(str(mask_path))
            mask_resized = cv2.resize(mask, (self.resolution, self.resolution), interpolation = cv2.INTER_NEAREST)
            mask_resized = mask_resized.astype(np.unit8) #png文件

            mask_png_path = self.annotations_folder / f"{img_id}.png"
            cv2.imwrite(str(mask_png_path), mask_resized)

            img_jpg_path = self.img_target_folder / f"{img_id}.jpg"
            cv2.imwrite(str(img_jpg_path), img_resized


