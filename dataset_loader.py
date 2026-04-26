# dataset_loader.py
import os
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from PIL import Image

# ========== 配置路径 ==========
COCO_ROOT = r"D:/PycharmProjects/large homework/data/coco"
VAL_IMG_DIR = os.path.join(COCO_ROOT, "val2017")
CAPTIONS_FILE = os.path.join(COCO_ROOT, "annotations", "captions_val2017.json")
# ================================================

def get_coco_validation_loader(batch_size=1, num_workers=0, safe_mode=True):
    """
    加载 COCO 验证集。
    参数:
        batch_size: 批次大小
        num_workers: 多进程数量
        safe_mode: 跳过缺失的图像文件
    返回:
        DataLoader 对象
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    if safe_mode:
        # 使用自定义的 SafeCocoCaptions 类，自动跳过缺失图片
        dataset = SafeCocoCaptions(
            root=VAL_IMG_DIR,
            annFile=CAPTIONS_FILE,
            transform=transform
        )
    else:
        # 使用官方类（要求数据集完整）
        dataset = dset.CocoCaptions(
            root=VAL_IMG_DIR,
            annFile=CAPTIONS_FILE,
            transform=transform
        )

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return loader


class SafeCocoCaptions(dset.CocoDetection):
    """
    如果某张图片缺失，会自动跳过并尝试下一张，直到找到有效图片。
    """
    def __init__(self, root, annFile, transform=None, max_attempts=100):
        super().__init__(root, annFile, transform=transform)
        self.max_attempts = max_attempts

    def __getitem__(self, index):
        attempts = 0
        while attempts < self.max_attempts:
            try:
                img, _ = super().__getitem__(index)
                img_id = self.ids[index]
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                captions = [ann['caption'] for ann in self.coco.loadAnns(ann_ids)]
                return img, captions
            except FileNotFoundError:
                # 跳过缺失的图片，尝试下一张
                index = (index + 1) % len(self.ids)
                attempts += 1
        raise FileNotFoundError(f"连续尝试 {self.max_attempts} 次均未找到有效图片，请检查数据集完整性。")