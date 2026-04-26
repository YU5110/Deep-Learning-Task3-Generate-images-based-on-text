# resize_coco.py
import os
from PIL import Image
from tqdm import tqdm

# ========== 配置路径 ==========
REAL_IMAGES_DIR = r"D:/PycharmProjects/large homework/data/coco/val2017"
RESIZED_DIR = r"D:/PycharmProjects/large homework/data/coco/val2017_resized_512"
TARGET_SIZE = (512, 512)


# =============================

def resize_all_images(src_folder, dst_folder, target_size):
    os.makedirs(dst_folder, exist_ok=True)
    files = [f for f in os.listdir(src_folder)
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"正在将 {len(files)} 张图像缩放至 {target_size}...")
    for filename in tqdm(files):
        src_path = os.path.join(src_folder, filename)
        dst_path = os.path.join(dst_folder, filename)
        try:
            with Image.open(src_path) as img:
                img_resized = img.resize(target_size, Image.LANCZOS)
                # 转换为 RGB 避免 RGBA 导致的问题
                if img_resized.mode != 'RGB':
                    img_resized = img_resized.convert('RGB')
                img_resized.save(dst_path)
        except Exception as e:
            print(f"处理 {filename} 时出错: {e}")


if __name__ == '__main__':
    resize_all_images(REAL_IMAGES_DIR, RESIZED_DIR, TARGET_SIZE)
    print(f"✅ 缩放完成！图像已保存至: {RESIZED_DIR}")