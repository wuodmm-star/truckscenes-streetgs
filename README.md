# Street Gaussians on MAN TruckScenes（卡车自动驾驶场景的动态 3DGS 重建）

> **English summary.** An adapter that brings the [MAN TruckScenes](https://registry.opendata.aws/man-truckscenes/) truck dataset
> (6 LiDARs, tractor + trailer, highway speeds) into [drivestudio](https://github.com/ziyc/drivestudio), plus Street
> Gaussians reconstructions of several mini-split scenes on a single 8 GB laptop GPU, and an analysis of where the
> method breaks: fast-moving vehicles on highways.

![scene 2 decomposition](assets/scene2_decomposition.gif)

*Scene 2（城区，低速）。从上到下：真实相机画面 / Street Gaussians 渲染 / 只渲染静态背景 / 只渲染车辆节点（随 3D 跟踪框移动）。
绿色 = 该部分没有内容（天空由单独的天空模型负责）。完整视频：[assets/scene2_decomposition.mp4](assets/scene2_decomposition.mp4)*

## 这个项目做了什么

1. **把卡车数据集接入 drivestudio**（drivestudio 原生只支持 Waymo / nuScenes / PandaSet 等乘用车数据集）
   - `addon/datasets/truckscenes/truckscenes_preprocess.py`：TruckScenes → drivestudio 场景格式
     - **6 个 LiDAR 融合**：每个 LiDAR 用各自时刻的自车位姿变换到世界坐标，再统一到参考帧，避免高速下多雷达之间的运动错位
     - **去掉自车点云**：牵引车 + 挂车会出现在 LiDAR 里，按自车坐标系的盒子剔除
     - **2 Hz 标注 → 10 Hz**：跟踪框位置线性插值、朝向球面插值（Slerp）
     - 新类别映射（`vehicle.other`、`vehicle.train` → 刚体；`vehicle.ego_trailer` 自己的挂车 → 不建模）
     - 每个场景输出 LiDAR + 跟踪框投影到相机的检查图，确认标定和时间同步正确
   - `addon/datasets/truckscenes/truckscenes_sourceloader.py`、`addon/configs/…`、`tools/patch_drivestudio.py`：注册数据集
2. **在 8 GB 显存笔记本（RTX 4060 + WSL2）上跑通 Street Gaussians**，环境脚本在 `scripts/`
3. **多场景批量实验 + 分析**：城区 / 高速、晴 / 阴、不同车速

## 结果

前 2 个相机（左前、右前），660×314，30k 步；每 8 个时间步留出 1 个作为测试帧（不参与训练）。

<!-- RESULTS_TABLE -->
| scene | area | weather | ego km/h | moving objects (median km/h) | PSNR ↑ | SSIM ↑ | LPIPS ↓ | vehicle PSNR | human PSNR |
|---|---|---|---|---|---|---|---|---|---|
| 2 | city | overcast | 21 | 48/111 (15) | 23.78 | 0.688 | 0.213 | 22.03 | 18.69 |
| 3 | highway | overcast | 86 | 76/81 (96) | 24.63 | 0.757 | 0.183 | 18.61 | – |
| 4 | highway | overcast | 63 | 91/95 (73) | 25.50 | 0.761 | 0.226 | 18.08 | – |
| 6 | city | clear | 26 | 49/126 (37) | 22.96 | 0.707 | 0.176 | 19.88 | 22.73 |
<!-- /RESULTS_TABLE -->

*场景 9、0、8 正在训练，完成后补充。夜间（1）、自车静止（5）、雪天（7）场景未使用。*

| 高速场景 3（测试帧） | 高速场景 4（测试帧） |
|---|---|
| ![](assets/scene3_test.gif) | ![](assets/scene4_test.gif) |

## 分析：高速场景里，车辆是最薄弱的部分

- **背景**：高速场景的整体 PSNR（24.6 / 25.5）反而高于城区。路面、护栏、天空结构简单，Street Gaussians 重建得很好。
- **车辆**：城区低速场景 2 的车辆 PSNR 为 22.0；两个高速场景只有 18.1–18.6。

![scene 4 vehicle](assets/scene4_vehicle_crop.jpg)

*场景 4 的测试帧：树的位置完全对齐，但卡车整体被渲染到偏前的位置，并且有重影。说明问题不在外观，而在**车辆在该时刻被放错了位置**。*

可能原因（按我的判断排序，尚未逐一做消融实验验证）：

1. **相机时刻与跟踪框时刻不一致**：图像的实际时间戳和 10 Hz 时间轴之间平均差 18 ms、最大约 65–98 ms（见预处理日志）。
   相机位姿用的是图像自己的时刻，但跟踪框是在时间轴时刻插值出来的。相对速度 100 km/h 时，65 ms 就是约 1.8 m 的位置误差。
   → **下一步**：按每张图像自己的时间戳插值跟踪框。这是本适配器可以直接改进的地方。
2. **2 Hz 标注的插值误差**：96 km/h 下，两次标注之间车辆移动约 13 m，变道 / 加减速时线性插值会偏离真实轨迹。
3. **Street Gaussians 本身依赖跟踪框**：它的框位姿微调只能修正小误差；而且远处车辆只占很少像素，监督信号弱。

## 复现

```bash
# 1. drivestudio + 环境（WSL2 Ubuntu / Linux，CUDA GPU；脚本按 8 GB 显存调过）
git clone https://github.com/ziyc/drivestudio && bash scripts/install_drivestudio.sh && bash scripts/install_segformer.sh
# 2. 安装 TruckScenes 适配器
bash tools/install_addon.sh /path/to/drivestudio
# 3. 下载 TruckScenes mini（v1.2-mini，需在官网同意使用协议），pip install truckscenes-devkit
# 4. 预处理 → 天空/动态遮罩 → 训练 + 评估，可中断后重跑
TS_ROOT=/path/to/man-truckscenes DS=/path/to/drivestudio bash scripts/run_batch.sh 2 3 4 6
# 5. 汇总结果表
python tools/collect_results.py --runs <OUT_ROOT>/truckscenes --data data/truckscenes/processed_10Hz/mini \
    --scene_json /path/to/man-truckscenes/v1.2-mini/scene.json --out results
```

单场景约 1.5 小时（RTX 4060 8 GB；WSL 内存限制 6 GB + 8 GB swap）。实践中遇到的坑：
- drivestudio 训练结束后的"全片渲染"会把所有帧放在内存里，高速长场景会耗尽内存 → `run_batch.sh` 里关掉了 `render_full` / `render_novel`，测试指标不受影响。
- CUDA 11.7 不认识 sm_89（RTX 40 系）→ 用 `TORCH_CUDA_ARCH_LIST=8.6` 编译，可以在 sm_89 上运行。

## 目录

```
addon/datasets/truckscenes/   预处理 + 数据加载（复制进 drivestudio/datasets/）
addon/configs/datasets/truckscenes/   2 相机 / 4 相机配置
tools/   安装适配器、注册数据集、结果汇总、测试帧导出、遮罩预览
scripts/ 环境安装、批量运行
results/ 结果表（csv / md）
assets/  动图与对比图
```

## 致谢与许可

- 方法与训练框架：[drivestudio](https://github.com/ziyc/drivestudio)（MIT, © 2024 Ziyu Chen），
  Street Gaussians（Yan et al., ECCV 2024）。本仓库只包含适配代码和脚本，不包含 drivestudio 源码。
- 数据：[MAN TruckScenes](https://arxiv.org/abs/2407.07462)，CC BY-NC-SA 4.0（仅限非商业用途）。
  本仓库不包含原始数据；`assets/` 中的图像来源于该数据集及基于其训练的渲染结果，按同一许可（CC BY-NC-SA 4.0）提供。
- 本仓库代码：MIT（见 [LICENSE](LICENSE)）。
