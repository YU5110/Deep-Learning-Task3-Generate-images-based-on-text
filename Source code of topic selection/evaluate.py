# evaluate.py
import torch
from pytorch_fid import fid_score

# ========== 修改后的路径 ==========
REAL_IMAGES_DIR = r"D:/PycharmProjects/large homework/data/coco/val2017_resized_512"  # 使用缩放后的文件夹
FAKE_IMAGES_DIR = r"D:/PycharmProjects/large homework/generated_images/sd_v1-5"


# ==================================

def calculate_fid(real_dir, fake_dir, batch_size=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"正在计算 FID（设备：{device}）...")

    fid = fid_score.calculate_fid_given_paths(
        [real_dir, fake_dir],
        batch_size=batch_size,
        device=device,
        dims=2048,
        num_workers=0  # 禁用多进程，避免 Windows 上的问题
    )
    print(f"✅ FID Score: {fid:.4f}")
    return fid


if __name__ == '__main__':
    print("=" * 50)
    print("开始评估生成图像质量")
    print("=" * 50)
    fid = calculate_fid(REAL_IMAGES_DIR, FAKE_IMAGES_DIR)
    print("\n评估完成！")