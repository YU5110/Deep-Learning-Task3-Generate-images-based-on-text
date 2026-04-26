import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

# 图像预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载训练集（这里只是定义，不会触发多进程）
train_dataset = dset.CocoCaptions(
    root='./data/coco/train2017',
    annFile='./data/coco/annotations/captions_train2017.json',
    transform=transform
)

# 创建 DataLoader（num_workers=4 表示使用4个子进程）
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# ✅ 主程序入口必须写在此判断内
if __name__ == '__main__':
    # 测试加载一个 batch
    images, captions = next(iter(train_loader))
    print(f"图像 batch 形状: {images.shape}")
    print(f"文本示例: {captions[0]}")