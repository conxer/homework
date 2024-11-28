import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn
from torch.optim.lr_scheduler import StepLR
from facades_dataset import FacadesDataset
from FCN_network import FullyConvNetwork, Discriminator
import numpy as np
import cv2

# 损失函数：生成器和判别器各自的损失
def criterion_GAN(prediction, target, is_real):
    if is_real:
        target = torch.ones_like(prediction)
    else:
        target = torch.zeros_like(prediction)
    return nn.BCELoss()(prediction, target)

def criterion_L1(output, target):
    return nn.L1Loss()(output, target)

def save_images(inputs, targets, outputs, folder_name, epoch, num_images=5):
    os.makedirs(f'{folder_name}/epoch_{epoch}', exist_ok=True)
    for i in range(num_images):
        input_img_np = tensor_to_image(inputs[i])
        target_img_np = tensor_to_image(targets[i])
        output_img_np = tensor_to_image(outputs[i])

        # 拼接并保存图像
        comparison = np.hstack((input_img_np, target_img_np, output_img_np))
        cv2.imwrite(f'{folder_name}/epoch_{epoch}/result_{i + 1}.png', comparison)

def tensor_to_image(tensor):
    image = tensor.cpu().detach().numpy()
    image = np.transpose(image, (1, 2, 0))  # 转换为 (H, W, C)
    image = (image + 1) / 2  # 归一化回 [0, 1] 范围
    image = (image * 255).astype(np.uint8)
    return image

def train_one_epoch(generator, discriminator, dataloader, optimizer_G, optimizer_D, device, epoch, num_epochs):
    generator.train()
    discriminator.train()
    
    running_loss_G = 0.0
    running_loss_D = 0.0

    for i, (image_rgb, image_semantic, input_image) in enumerate(dataloader):
        image_rgb = image_rgb.to(device)
        image_semantic = image_semantic.to(device)
        input_image = input_image.to(device)

        # ----------------------
        # 训练判别器
        # ----------------------
        optimizer_D.zero_grad()

        # 判别器判断真实图像
        real_pred = discriminator(input_image)
        loss_D_real = criterion_GAN(real_pred, image_semantic, True)

        # 判别器判断生成图像
        fake_image = generator(image_rgb)
        fake_input_image = torch.cat([image_rgb, fake_image], dim=1)  # 拼接通道维度，dim=1
        fake_pred = discriminator(fake_input_image)
        loss_D_fake = criterion_GAN(fake_pred, image_semantic, False)

        # 总判别器损失
        loss_D = (loss_D_real + loss_D_fake) / 2
        loss_D.backward(retain_graph=True)  # 保留计算图，避免第二次反向传播报错
        optimizer_D.step()

        # ----------------------
        # 训练生成器
        # ----------------------
        optimizer_G.zero_grad()

        # 生成器损失
        fake_pred = discriminator(fake_input_image)
        loss_G = criterion_GAN(fake_pred, image_semantic, True) + criterion_L1(fake_image, image_semantic) * 100

        loss_G.backward()  # 计算生成器的梯度
        optimizer_G.step()

        running_loss_G += loss_G.item()
        running_loss_D += loss_D.item()

        if i % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss_G: {loss_G.item():.4f}, Loss_D: {loss_D.item():.4f}')

    # 在每个 epoch 结束后保存生成的图像
    save_images(image_rgb, image_semantic, fake_image, 'generated_images', epoch)

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 创建 checkpoints 文件夹
    os.makedirs('checkpoints', exist_ok=True)

    # 初始化数据集和数据加载器
    train_dataset = FacadesDataset(list_file='train_list.txt')
    val_dataset = FacadesDataset(list_file='val_list.txt')
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    # 初始化模型、损失函数和优化器
    generator = FullyConvNetwork().to(device)
    discriminator = Discriminator().to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    num_epochs = 200
    for epoch in range(num_epochs):
        train_one_epoch(generator, discriminator, train_loader, optimizer_G, optimizer_D, device, epoch, num_epochs)

        # 保存模型检查点
        if (epoch + 1) % 20 == 0:
            torch.save(generator.state_dict(), f'checkpoints/generator_epoch_{epoch+1}.pth')
            torch.save(discriminator.state_dict(), f'checkpoints/discriminator_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    main()
