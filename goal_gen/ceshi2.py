import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from transformers import T5Tokenizer, T5EncoderModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from utils.pipeline import Pipeline

class IP2PImageEditor:
    def __init__(self):
        # 初始化配置参数
        self.res = 128
        self.ckpt_path = "/openbayes/home/GR-MG/goal_gen/checkpoint/2025-04-28/train_goal_gen/epoch=9-step=81650.ckpt"
        self.pretrained_model_dir = "/openbayes/home/GR-MG/resources/IP2P/instruct-pix2pix"
        
        # 初始化模型组件
        self._init_models()
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((self.res, self.res)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        # 推理参数
        self.generator = torch.Generator("cuda").manual_seed(42)
        self.num_inference_steps = 50
        self.image_guidance_scale = 2.5
        self.guidance_scale = 7.5

    def _init_models(self):
        """初始化所有模型组件"""
        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
        self.text_encoder = T5EncoderModel.from_pretrained("t5-base")
        self.vae = AutoencoderKL.from_pretrained(self.pretrained_model_dir, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(self.pretrained_model_dir, subfolder="unet")

        # 加载预训练权重
        payload = torch.load(self.ckpt_path)
        state_dict = payload['state_dict']
        del payload
        msg = self.unet.load_state_dict(state_dict['unet'], strict=True)
        print(f"Model loading status: {msg}")

        # 构建推理管道
        self.pipe = Pipeline.from_pretrained(
            self.pretrained_model_dir,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            vae=self.vae,
            unet=self.unet,
            torch_dtype=torch.bfloat16
        ).to("cuda")
        self.pipe.safety_checker = None
        self.pipe.requires_safety_checker = False

    def edit_image(self, input_image, instruction, output_path="output.jpg"):
        """
        处理图片编辑请求
        :param input_image: PIL.Image对象或图片路径
        :param instruction: 文本编辑指令
        :param output_path: 输出图片保存路径
        :return: 编辑后的PIL.Image对象
        """
        # 加载输入图片
        if isinstance(input_image, str):
            input_image = Image.open(input_image).convert("RGB")
        
        # 预处理
        input_tensor = self.transform(input_image).unsqueeze(0).to("cuda")
        
        # 执行推理
        edited_images = self.pipe(
            prompt=[instruction],
            image=input_tensor,
            num_inference_steps=self.num_inference_steps,
            image_guidance_scale=self.image_guidance_scale,
            guidance_scale=self.guidance_scale,
            generator=self.generator
        ).images

        # 保存结果
        edited_images[0].save(output_path)
        print(f"Edited image saved to {output_path}")
        return edited_images[0]

    def batch_process_images(self, input_base_dir, output_base_dir, instruction, episode_range=range(10)):
        """
        批量处理多个episode目录中的图片
        :param input_base_dir: 输入基础目录（包含episode_n子目录）
        :param output_base_dir: 输出基础目录
        :param instruction: 统一的文本编辑指令
        :param episode_range: episode编号范围，默认为0-9
        """
        # 确保输出基础目录存在
        if not os.path.exists(output_base_dir):
            os.makedirs(output_base_dir)
            print(f"Created output base directory: {output_base_dir}")
        
        # 处理每个episode目录
        for n in episode_range:
            input_dir = os.path.join(input_base_dir, f"episode{n}")
            output_dir = os.path.join(output_base_dir, f"episode{n}")
            
            # 确保输出子目录存在
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Created output subdirectory: {output_dir}")
            
            # 获取当前episode目录中的所有图片文件
            try:
                image_files = [f for f in os.listdir(input_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            except FileNotFoundError:
                print(f"Warning: Input directory not found: {input_dir}")
                continue
            
            print(f"Processing episode {n}: found {len(image_files)} images")
            
            # 批量处理当前episode中的每张图片
            for filename in image_files:
                try:
                    input_path = os.path.join(input_dir, filename)
                    output_path = os.path.join(output_dir, filename)
                    
                    print(f"Processing: {filename}")
                    self.edit_image(
                        input_image=input_path,
                        instruction=instruction,
                        output_path=output_path
                    )
                    
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    continue

# 示例用法
if __name__ == "__main__":
    # 初始化编辑器
    editor = IP2PImageEditor()
    
    # 批量处理图片
    input_base_dir = "/openbayes/home/GR-MG/resources/ndata/blocks_stack_easy"
    output_base_dir = "/openbayes/home/GR-MG/resources/ceshi/shuchu5"
    instruction = "stack blocks"
    
    editor.batch_process_images(
        input_base_dir=input_base_dir,
        output_base_dir=output_base_dir,
        instruction=instruction,
        episode_range=range(10)  # 处理episode_0到episode_9
    )