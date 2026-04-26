# evaluate_100.py
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from pytorch_fid import fid_score
from PIL import Image
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel

# ========== 配置路径 ==========
REAL_IMAGES_DIR = r"D:/PycharmProjects/large homework/data/coco/val2017_resized_512"
FAKE_IMAGES_DIR = r"D:/PycharmProjects/large homework/generated_images/sd_v1-5_100"
TEXT_DIR = r"D:/PycharmProjects/large homework/generated_images/sd_v1-5_100_texts"


# ==================================

def calculate_fid(real_dir, fake_dir, batch_size=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"正在计算 FID（设备：{device}）...")
    fid = fid_score.calculate_fid_given_paths(
        [real_dir, fake_dir],
        batch_size=batch_size,
        device=device,
        dims=2048,
        num_workers=0
    )
    print(f"✅ FID Score: {fid:.4f}")
    return fid


def prepare_text_files_for_clip_score(prompts, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, prompt in enumerate(prompts):
        filename = f"prompt_{i:04d}_sample_00.txt"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(prompt)
    print(f"已准备 {len(prompts)} 个文本文件到 {output_dir}")


def calculate_clip_score(image_dir, text_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 使用 transformers 加载 CLIP 模型和处理器
    print("正在加载 CLIP 模型...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    scores = []
    img_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    if not img_files:
        print("警告：图像目录中没有找到任何图像文件！")
        return 0.0

    print(f"正在计算 CLIP Score (共 {len(img_files)} 张图像)...")

    # 批量处理可以提高速度，这里逐张计算，简单清晰
    for img_file in img_files:
        img_path = os.path.join(image_dir, img_file)
        txt_file = os.path.splitext(img_file)[0] + ".txt"
        txt_path = os.path.join(text_dir, txt_file)

        if not os.path.exists(txt_path):
            print(f"警告：未找到 {img_file} 对应的文本文件，跳过。")
            continue

        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()

        img = Image.open(img_path).convert('RGB')

        # 使用 processor 处理图像和文本
        inputs = processor(text=[text], images=img, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # 获取图像和文本的特征向量
            image_embeds = outputs.image_embeds  # [1, 512]
            text_embeds = outputs.text_embeds  # [1, 512]

            # 归一化
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

            # 计算余弦相似度
            score = (image_embeds * text_embeds).sum(dim=-1).item()
            scores.append(score)

    if not scores:
        print("警告：没有计算到任何有效的 CLIP Score！")
        return 0.0

    avg_score = sum(scores) / len(scores)
    print(f"✅ CLIP Score: {avg_score:.4f} (共 {len(scores)} 张图片)")
    return avg_score


if __name__ == '__main__':
    print("=" * 50)
    print("开始评估生成图像质量（100张样本）")
    print("=" * 50)

    # 1. 计算 FID
    fid = calculate_fid(REAL_IMAGES_DIR, FAKE_IMAGES_DIR)

    # 2. 准备文本文件并计算 CLIP Score
    print("\n正在准备 CLIP Score 所需的文本文件...")
    from dataset_loader import get_coco_validation_loader

    val_loader = get_coco_validation_loader(batch_size=1, num_workers=0, safe_mode=True)
    prompts = []
    for images, captions in val_loader:
        first_caption = captions[0][0]
        prompts.append(first_caption)
        if len(prompts) >= 100:
            break
    prepare_text_files_for_clip_score(prompts, TEXT_DIR)

    print("\n正在计算 CLIP Score...")
    clip = calculate_clip_score(FAKE_IMAGES_DIR, TEXT_DIR)

    print("\n" + "=" * 50)
    print(f"FID: {fid:.4f}")
    print(f"CLIP Score: {clip:.4f}")
    print("=" * 50)
    print("评估完成！")