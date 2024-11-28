import torch
from torch.utils.data import Dataset
import cv2

class FacadesDataset(Dataset):
    def __init__(self, list_file):
        # 读取图像文件路径列表
        with open(list_file, 'r') as file:
            self.image_filenames = [line.strip() for line in file]
        
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        # 获取图像路径
        img_name = self.image_filenames[idx]
        
        # 读取图像
        img_color_semantic = cv2.imread(img_name)
        img_color_semantic = cv2.cvtColor(img_color_semantic, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式
        
        # 调整图像大小至 1024x512
        img_color_semantic = cv2.resize(img_color_semantic, (1024, 512))
        
        # 将图像转换为 PyTorch tensor，并归一化到 [-1, 1] 范围
        image = torch.from_numpy(img_color_semantic).permute(2, 0, 1).float() / 255.0 * 2.0 - 1.0
        
        # 根据 1024x512 分辨率切分图像
        image_rgb = image[:, :, :512]  # 前 512 列为输入图像
        image_semantic = image[:, :, 512:]  # 后 512 列为目标图像
        
        # 拼接图像作为判别器的输入
        # 判别器输入是由真实图像和生成图像拼接而成的，所以返回拼接后的图像
        input_image = torch.cat([image_rgb, image_semantic], dim=0)  # [C, H, W] -> [6, H, W]
        
        return image_rgb, image_semantic, input_image
