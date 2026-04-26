import os
from PIL import Image

def check_image_sizes(folder_path):
    print(f"正在检查文件夹: {folder_path}")
    sizes = {}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(folder_path, filename)
            try:
                with Image.open(filepath) as img:
                    size = img.size  # (width, height)
                    sizes[size] = sizes.get(size, 0) + 1
            except Exception as e:
                print(f"无法读取图像 {filename}: {e}")

    if len(sizes) > 1:
        print("警告：发现多种尺寸！")
        for size, count in sizes.items():
            print(f"  尺寸 {size}: {count} 张")
    else:
        print(f"所有图像尺寸一致: {list(sizes.keys())[0]}")
    print("-" * 30)

# 实际路径
REAL_IMAGES_DIR = r"D:/PycharmProjects/large homework/data/coco/val2017"
FAKE_IMAGES_DIR = r"D:/PycharmProjects/large homework/generated_images/sd_v1-5"

check_image_sizes(REAL_IMAGES_DIR)
check_image_sizes(FAKE_IMAGES_DIR)