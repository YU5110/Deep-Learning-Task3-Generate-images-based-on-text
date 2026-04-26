# generate_100.py
import os
import torch
from diffusers import StableDiffusionPipeline
from dataset_loader import get_coco_validation_loader
from tqdm import tqdm

# ========== 配置参数 ==========
# 关键修改：使用本地模型路径，不再联网下载
MODEL_PATH = r"C:\Users\YU\.cache\huggingface\hub\models--runwayml--stable-diffusion-v1-5\snapshots\451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
OUTPUT_DIR = r"D:/PycharmProjects/large homework/generated_images/sd_v1-5_100"
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 7.5
BATCH_SIZE = 2
NUM_PER_PROMPT = 1
# =============================

def generate_images_from_prompts(prompts, pipe, output_dir, num_per_prompt=1):
    os.makedirs(output_dir, exist_ok=True)
    generated_paths = []
    total = len(prompts) * num_per_prompt
    pbar = tqdm(total=total, desc="生成图像")
    for i, prompt in enumerate(prompts):
        for j in range(num_per_prompt):
            with torch.autocast("cuda"):
                image = pipe(
                    prompt=prompt,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                ).images[0]
            filename = f"prompt_{i:04d}_sample_{j:02d}.png"
            filepath = os.path.join(output_dir, filename)
            image.save(filepath)
            generated_paths.append(filepath)
            pbar.update(1)
    pbar.close()
    return generated_paths


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 加载模型（不变）
    print("正在加载 Stable Diffusion 模型...")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_PATH,  # 请确保 MODEL_PATH 指向本地模型
        torch_dtype=torch.float16,
        safety_checker=None,
        local_files_only=True
    )
    pipe = pipe.to("cuda")

    # 使用新版 API 启用 VAE 切片 (降低峰值显存)
    pipe.vae.enable_slicing()
    # 启用注意力切片 (进一步降低显存，对速度影响小)
    pipe.enable_attention_slicing()
    print("✅ 优化配置完成 (使用 PyTorch 2.0 原生 SDPA 加速)")

    # 2. 加载 COCO 验证集，获取所有文本描述
    print("正在加载 COCO 验证集所有描述...")
    val_loader = get_coco_validation_loader(batch_size=1, num_workers=0, safe_mode=True)

    print("正在检查 val_loader 返回的数据结构...")
    val_loader = get_coco_validation_loader(batch_size=1, num_workers=0, safe_mode=True)

    for i, (images, captions) in enumerate(val_loader):
        if i == 0:  # 只看第一个 batch
            print(f"captions 的类型: {type(captions)}")
            print(f"captions 的长度: {len(captions)}")
            print(f"captions[0] 的类型: {type(captions[0])}")
            print(f"captions[0] 的内容: {captions[0]}")
            break

    all_prompts = []
    for images, captions in val_loader:
        # captions 是一个长度为 5 的列表，每个元素是 tuple
        first_caption_tuple = captions[0]  # 取第一个 tuple
        first_caption = first_caption_tuple[0]  # 取 tuple 中的字符串
        all_prompts.append(first_caption)

        #限制生成的图片数量（例如只生成前1000张）
        if len(all_prompts) >= 100:
            break

    print(f"共获取到 {len(all_prompts)} 条文本描述。")

    # 3. 批量生成图像
    print("开始批量生成图像...")
    generated_paths = generate_images_from_prompts(
        prompts=all_prompts,  # 传入所有 prompts
        pipe=pipe,
        output_dir=OUTPUT_DIR,
        num_per_prompt=NUM_PER_PROMPT
    )
    print(f"✅ 生成完成！共生成 {len(generated_paths)} 张图像，保存在 {OUTPUT_DIR}")