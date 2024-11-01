import torch
from torch.utils.data import Dataset
import cv2

class FacadesDataset(Dataset):
    def __init__(self, list_file):
        with open(list_file, 'r') as file:
            self.image_filenames = [line.strip() for line in file]
        
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_color_semantic = cv2.imread(img_name)
        img_color_semantic = cv2.cvtColor(img_color_semantic, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式
        img_color_semantic = cv2.resize(img_color_semantic, (1024, 512))
        image = torch.from_numpy(img_color_semantic).permute(2, 0, 1).float() / 255.0 * 2.0 - 1.0
        
        # 根据 1024x512 分辨率切分
        image_rgb = image[:, :, :512]  # 前 512 列
        image_semantic = image[:, :, 512:]  # 后 512 列
        
        return image_rgb, image_semantic
