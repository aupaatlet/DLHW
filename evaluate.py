import os
import numpy as np
from skimage import io, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.transform import resize

FOLDER_CONFIG = [
    {
        "name": "blocks_stack_easy",
        "real_root": "/Users/liyiqi/Desktop/blocks_stack_easy/real",
        "gen_root": "/Users/liyiqi/Desktop/blocks_stack_easy/generated"
    },
    {
        "name": "block_hammer_beat",
        "real_root": "/Users/liyiqi/Desktop/block_hammer_beat/real",
        "gen_root": "/Users/liyiqi/Desktop/block_hammer_beat/generated"
    },
    {
        "name": "block_handover",
        "real_root": "/Users/liyiqi/Desktop/block_handover/real",
        "gen_root": "/Users/liyiqi/Desktop/block_handover/generated"
    }
]


def process_episode(real_dir, gen_dir, episode_num):
    """处理单个episode文件夹"""
    ssim_list = []
    psnr_list = []

    # 处理10对对比 (real: 002-011 vs gen: 000-009)
    for pair_num in range(10):
        real_id = f"{(pair_num + 2):03d}"  # 真实图像编号 002-011
        gen_id = f"{pair_num:03d}"  # 生成图像编号 000-009

        real_path = os.path.join(real_dir, f"image_{real_id}.png")
        gen_path = os.path.join(gen_dir, f"image_{gen_id}.png")

        # 验证文件存在
        if not os.path.exists(real_path):
            print(f"缺失真实图像: {real_path}")
            continue
        if not os.path.exists(gen_path):
            print(f"缺失生成图像: {gen_path}")
            continue

        try:
            # 读取和处理图像
            real_img = img_as_float(io.imread(real_path))
            gen_img = img_as_float(io.imread(gen_path))

            # 统一尺寸为128x128
            real_resized = resize(real_img, (128, 128, 3), anti_aliasing=True)

            # 尺寸验证
            if real_resized.shape != gen_img.shape:
                print(f"尺寸不匹配 [episode{episode_num}]: {real_id} vs {gen_id}")
                continue

            # 计算指标
            ssim_val = ssim(real_resized, gen_img,
                            channel_axis=-1,
                            data_range=gen_img.max() - gen_img.min())
            psnr_val = psnr(real_resized, gen_img,
                            data_range=gen_img.max() - gen_img.min())

            ssim_list.append(ssim_val)
            psnr_list.append(psnr_val)

            print(f"Episode{episode_num}|{pair_num + 1}/10: SSIM {ssim_val:.4f} PSNR {psnr_val:.2f}dB")

        except Exception as e:
            print(f"处理失败 [episode{episode_num}]: {str(e)}")

    # 计算单个episode平均值
    avg_ssim = np.mean(ssim_list) if ssim_list else 0
    avg_psnr = np.mean(psnr_list) if psnr_list else 0
    return avg_ssim, avg_psnr, len(ssim_list)


def process_task(task_config):
    """处理整个大任务文件夹（包含多个episode）"""
    task_name = task_config["name"]
    all_episode_ssim = []
    all_episode_psnr = []

    print(f"\n{'#' * 40}")
    print(f"开始处理任务: {task_name}")
    print(f"真实目录: {task_config['real_root']}")
    print(f"生成目录: {task_config['gen_root']}")

    # 遍历所有episode（假设有10个episode）校验:0-9
    for episode_num in range(10):
        episode_name = f"episode{episode_num}"

        # 构建完整路径
        real_dir = os.path.join(task_config["real_root"], episode_name)
        gen_dir = os.path.join(task_config["gen_root"], episode_name)

        # 验证episode存在
        if not os.path.exists(real_dir):
            print(f"缺失真实目录: {real_dir}")
            continue
        if not os.path.exists(gen_dir):
            print(f"缺失生成目录: {gen_dir}")
            continue

        # 处理单个episode
        avg_ssim, avg_psnr, valid_pairs = process_episode(
            real_dir=real_dir,
            gen_dir=gen_dir,
            episode_num=episode_num
        )

        # 记录结果
        if valid_pairs > 0:
            all_episode_ssim.append(avg_ssim)
            all_episode_psnr.append(avg_psnr)
            print(f"Episode{episode_num} 完成 {valid_pairs}/10 对比")

    # 计算任务全局平均
    if all_episode_ssim:
        task_avg_ssim = np.mean(all_episode_ssim)
        task_avg_psnr = np.mean(all_episode_psnr)
        print(f"\n任务 {task_name} 完成！共处理 {len(all_episode_ssim)} 个有效episode")
        print(f"全局 SSIM: {task_avg_ssim:.4f}  全局 PSNR: {task_avg_psnr:.2f}dB")
    else:
        print(f"\n警告：任务 {task_name} 无有效数据")
        task_avg_ssim, task_avg_psnr = 0, 0

    return task_avg_ssim, task_avg_psnr


# 主程序
if __name__ == "__main__":
    final_results = []

    for task in FOLDER_CONFIG:
        ssim_val, psnr_val = process_task(task)
        final_results.append((task["name"], ssim_val, psnr_val))

    # 打印汇总报表
    print("\n\n{'='*40}")
    print(f"{'任务名称':<20} | {'平均SSIM':^8} | {'平均PSNR':^8}")
    print("-" * 42)
    for name, ssim, psnr in final_results:
        print(f"{name:<20} | {ssim:.4f}    | {psnr:.2f} dB")
    print("=" * 40)