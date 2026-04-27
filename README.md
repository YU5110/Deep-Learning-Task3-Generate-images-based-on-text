# 基于 Stable Diffusion 的文生图效果评估 (Text-to-Image Evaluation)

[![GitHub](https://img.shields.io/badge/GitHub-Project-blue?logo=github)](https://github.com/YU5110/Deep-Learning-Task3-Generate-images-based-on-text)

本项目是针对生成式人工智能课程设计的实验项目。基于 **Stable Diffusion v1.5** 模型，我们搭建了一套完整的“数据预处理 - 本地模型推理 - 客观量化评估”实验流水线（Pipeline），旨在验证大模型在复杂语义驱动下的视觉生成能力。

## 📖 项目简介
本项目以权威的 **MS-COCO 2017 验证集** 为基础，通过对 500 组高质量图文配对样本的批量推理，评估了 Stable Diffusion 模型在图像感知质量（FID）与图文对齐度（CLIP Score）两个维度的表现。针对本地算力受限（RTX 5070 Ti Laptop）的挑战，我们成功实践了 VAE Slicing 与 Attention Slicing 等显存优化策略。

## ✨ 核心亮点
- **工程化流水线**：打通了从 COCO 数据集自动化裁剪、安全加载到模型批量生成的全过程。
- **显存深度优化**：通过 FP16 半精度推理与注意力切片技术，在 12GB 显存设备上稳定完成长程批量任务。
- **学术化量化评估**：严谨引入 FID 与 CLIP Score 指标，并结合实验数据对指标在不同样本规模下的局限性进行了深度剖析。

## 📊 实验结果 (Quantitative Results)
| 评估指标 | 实验测得数值 | 理想范围参考 | 表现评价 |
| :--- | :--- | :--- | :--- |
| **FID Score** | 84.7789 | 通常 < 20 | 数值偏高，主要受制于 500 张样本的统计学偏差 |
| **CLIP Score** | 0.3119 | 通常 > 0.25 | 图文语义一致性表现优异，对齐度高 |

## 🛠️ 环境配置
### 硬件设备
- **GPU**: NVIDIA GeForce RTX 5070 Ti Laptop GPU (12GB 独立显存)
- **内存**: 32GB RAM

### 软件环境
- **框架**: PyTorch 2.11.0+cu128, Diffusers, Transformers
- **加速**: xFormers / PyTorch 2.0 SDPA

## 📂 仓库目录说明
```text
├── src/                          # 源代码
│   ├── dataset_loader.py         # 高鲁棒性 COCO 数据加载器
│   ├── resize_coco.py            # 图像缩放与中心裁剪预处理
│   ├── generate_500.py           # SD 推理主程序与显存优化部署
│   └── evaluate_500.py           # FID 与 CLIP Score 评估计算
├── docs/                         # 文档与成果展示
│   └── 课程报告.pdf               # 学术报告
│   
├── .gitignore                    # 忽略大型权重与数据集
└── README.md                     # 项目说明文档
```

## 👨‍💻 小组分工
- **王亦凡**：负责数据收集与预处理逻辑编写、量化评估脚本开发、数据分析。
- **李鑫予**：负责本地模型推理实现及显存优化策略部署，主导答辩汇报。
- **任龙全**：负责模型架构梳理、PPT设计、结课报告统筹撰写与排版。

## 📚 参考文献
1. Rombach R, et al. High-resolution image synthesis with latent diffusion models. CVPR 2022。
2. Radford A, et al. Learning transferable visual models from natural language supervision. ICML 2021。
3. Heusel M, et al. GANs trained by a two time-scale update rule converge to a local Nash equilibrium. NeurIPS 2017。
