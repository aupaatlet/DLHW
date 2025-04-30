import os
import glob
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from tqdm import tqdm

class MultiTaskImageDataset(Dataset):
    def __init__(self,
                 data_dir,  # 根目录，包含三个任务文件夹
                 resolution=128,
                 resolution_before_crop=160,
                 center_crop=False,
                 frame_offset=2,  # 固定偏移2帧
                 is_training=True,
                 color_aug=False):
        super().__init__()
        self.is_training = is_training
        self.color_aug = color_aug
        self.center_crop = center_crop
        self.data_dir = data_dir
        self.frame_offset = frame_offset
        self.resolution = resolution
        self.resolution_before_crop = resolution_before_crop

        # 定义三个任务及其对应的文本描述
        self.tasks = {
            "block_hammer_beat": "beat the block with the hammer",
            "block_handover": "handover the blocks",
            "blocks_stack_easy": "stack blocks"
        }

        # 收集所有有效的样本对（输入图像路径，输出图像路径，动作文本）
        self.samples = self._collect_samples()

        # 图像预处理
        if self.is_training:
            self.transform = transforms.Compose([
                transforms.Resize((self.resolution_before_crop, self.resolution_before_crop)),
                transforms.CenterCrop(self.resolution) if self.center_crop else transforms.RandomCrop(self.resolution),
            ])
        else:
            self.transform = transforms.Resize((self.resolution, self.resolution))
        
        print("=" * 20)
        print(f'共 {len(self)} 个有效样本对...')
    
    def _collect_samples(self):
        samples = []
        
        # 遍历每个任务文件夹
        for task_name, action_text in self.tasks.items():
            task_dir = os.path.join(self.data_dir, task_name)
            
            # 遍历每个episode文件夹（0到99）
            for episode in range(100):
                episode_dir = os.path.join(task_dir, f"episode{episode}")
                
                if not os.path.exists(episode_dir):
                    continue
                
                # 获取并排序当前episode的所有图片
                image_files = sorted(glob.glob(os.path.join(episode_dir, "image_*.png")),
                                   key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
                
                # 为当前episode创建有效样本对
                for i in range(len(image_files) - self.frame_offset):
                    input_path = image_files[i]
                    output_path = image_files[i + self.frame_offset]
                    samples.append((input_path, output_path, action_text))
        
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if not self.is_training:
            np.random.seed(index)
            random.seed(index)
        
        input_path, output_path, action_text = self.samples[index]

        # 加载输入和输出图像
        input_image = Image.open(input_path).convert("RGB")
        output_image = Image.open(output_path).convert("RGB")

        # 颜色增强
        if self.is_training and self.color_aug and random.random() > 0.4:
            bright_range = random.uniform(0.8, 1.2)
            contrast_range = random.uniform(0.8, 1.2)
            saturation_range = random.uniform(0.8, 1.2)
            hue_range = random.uniform(-0.04, 0.04)
            
            color_trans = transforms.ColorJitter(
                brightness=(bright_range, bright_range),
                contrast=(contrast_range, contrast_range),
                saturation=(saturation_range, saturation_range),
                hue=(hue_range, hue_range))
            
            input_image = color_trans(input_image)
            output_image = color_trans(output_image)
            
        # 合并并预处理图像
        concat_images = np.concatenate([np.array(input_image), np.array(output_image)], axis=2)
        concat_images = torch.tensor(concat_images)
        concat_images = concat_images.permute(2, 0, 1)
        concat_images = 2 * (concat_images / 255) - 1

        concat_images = self.transform(concat_images)
        input_image, output_image = concat_images.chunk(2)

        example = {
            'input_text': [action_text],
            'original_pixel_values': input_image.reshape(3, self.resolution, self.resolution),
            'edited_pixel_values': output_image.reshape(3, self.resolution, self.resolution),
            'progress': int((index + 1) / len(self) * 10)  # 计算进度
        }
        return example


if __name__ == "__main__":
    # 示例用法
    dataset = MultiTaskImageDataset(
        data_dir="/openbayes/home/GR-MG/resources/data",
        resolution=128,
        resolution_before_crop=160,
        center_crop=True,
        frame_offset=2,  # 固定偏移2帧
        is_training=True,
        color_aug=True
    )
    
    # 测试第一个样本
    example = dataset[0]
    print(f"输入文本: {example['input_text']}")
    print(f"输入图片形状: {example['original_pixel_values'].shape}")
    print(f"目标图片形状: {example['edited_pixel_values'].shape}")