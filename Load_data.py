from torch.utils.data import DataLoader
from Convert import Convert

source_img_folder = "path/to/source/image"
source_mask_folder = "paht/to/source/mask"
target_folder = "paht/to/target/folder"

dataset = Convert(source_img_folder, source_mask_folder, target_folder)

dataloader = DataLoader(dataset, batch_size = 1, shuffle = True)