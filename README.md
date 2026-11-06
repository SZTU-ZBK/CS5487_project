# CS5487：手写数字分类

三个 notebook 分别完成一个方法。数据读取、CNN 训练、评估和绘图放在普通 `.py` 文件中，不使用复杂框架。

## 环境和启动

本机环境名：`cs5487-digits`，Python 3.11。

```powershell
conda activate cs5487-digits
python -m jupyterlab
```

在 Jupyter 或 VS Code 中选择 **Python (cs5487-digits)** 内核，按顺序运行：

1. `01_1nn.ipynb`：读取数据、数据可视化、两个 trial 的 1-NN 基线。
2. `02_cnn.ipynb`：增强预览、CNN 训练、普通几何增强与精选增强的两组对照。
3. `03_scattering.ipynb`：小波散射特征、线性 SVM、验证集参数选择、评估。

如果 PowerShell 提示找不到 `conda`，本机也可以直接运行：

```powershell
& "$env:USERPROFILE\miniconda3\envs\cs5487-digits\python.exe" -m jupyterlab
```

也可以一次执行并保存 notebook 输出及 HTML 版本：

```powershell
python run_notebooks.py
# 或者只运行一个
python run_notebooks.py 02_cnn.ipynb
```

在新机器上重建环境：

```powershell
conda env create -f environment.yml
conda activate cs5487-digits
python -m ipykernel install --user --name cs5487-digits --display-name "Python (cs5487-digits)"
```

`requirements-lock.txt` 记录本次实际安装的版本；需要完全一致的依赖版本时，可在 Python 3.11 环境中运行 `python -m pip install -r requirements-lock.txt`。Windows 下无需 GPU；本项目使用 CPU 版 PyTorch，不依赖 CUDA。CPU 训练时间取决于机器性能。

## 文件说明

| 文件 | 内容 |
|---|---|
| `data_utils.py` | 直接从 zip 读取数据，恢复图像，转换 MATLAB 索引，检查划分 |
| `plot_utils.py` | 数据图、验证集划分图、训练曲线，统一导出 PNG/PDF |
| `evaluation.py` | 指标、预测表、混淆矩阵、错误案例和跨方法比较 |
| `cnn_utils.py` | 两层卷积 CNN、图像增强、训练循环、保存及加载模型 |
| `augmentation.py` | 仿射变换、弹性形变、腐蚀／膨胀、模糊及亮度／对比度变换 |
| `cnn_config.json` | 两组最终 CNN 设置；只记录入选方案，不保存候选实验结果表 |
| `scattering_utils.py` | 分批提取并缓存固定小波散射特征，绘制特征图 |
| `run_notebooks.py` | 顺序执行 notebook，保存输出和 HTML |
| `CS5487_Group34_proposal.pdf` | 已填写三位组员姓名、学号的一页 proposal 提交版 |
| `proposal.md` / `proposal.docx` | Proposal 的可编辑源文件 |
| `PA-3-courseproject.pdf` / `PA3_Translation.pdf` | 原始作业说明及译文 |

原始 `.zip` 文件保持不变，不需要手动解压。

## Git 版本管理

仓库保留源代码、已执行的三个 notebook、环境配置、作业说明、原始数据压缩包、proposal，以及 `results/` 中的 CSV、JSON 和 PNG/PDF 图表，方便直接查看实验记录和准备报告。

模型权重（`.pt`、`.joblib`）、二进制数组（`.npy`、`.npz`）、HTML 导出、特征缓存、Jupyter 运行文件和 Python 缓存不纳入提交。这些文件会保留在本地，也可以按上述命令重跑 notebook 生成。文本源文件统一使用 LF 换行。

## 本次已运行的结果

使用本机 CPU、固定种子及 `cnn_config.json` 的最终配置，完成作业规定的两个 trial。CNN 只保留普通几何增强和精选增强两组：

| 方法／设置 | Trial 1 | Trial 2 | 平均准确率 |
|---|---:|---:|---:|
| 1-NN | 91.35% | 91.85% | 91.600% |
| CNN：普通几何增强 | 97.05% | 96.05% | 96.550% |
| CNN：精选增强 | 97.10% | 97.20% | 97.150% |
| Wavelet Scattering + 线性 SVM | 97.65% | 97.55% | 97.600% |

本轮精选增强相对普通几何增强在两个 trial 中分别改变 +0.05、+1.15 个百分点，平均改变 +0.60 个百分点。

候选筛选只使用各 trial 自己的训练内部验证集，测试前冻结最终方案。这份课程数据的测试结果在先前开发时已查看过，因此本轮属于迭代实验，不称作新的独立盲测。当前只运行一个随机种子，不据此声称统计显著或保证附加挑战成绩。

最终入选的额外增强如下（概率表示每张训练图像被应用的概率）：

- Trial 1：弹性形变：位移 RMS 1.0 像素，平滑 σ=4，概率 60%。
- Trial 2：弹性形变：位移 RMS 0.7 像素，平滑 σ=4，概率 60%；高斯模糊 σ=0.2–0.7，概率 15%；亮度／对比度 ±15%，概率 20%。

报告／PPT 可优先使用：

- `results/comparison/accuracy_comparison.png`：方法准确率比较。
- `results/comparison/cnn_augmentation_ablation.png`：精选增强相对几何增强的收益。
- `results/comparison/cnn_learning_curves.png`：最终两组的验证 loss／准确率对照。
- CNN 各 trial 的 `learning_curves.png`：完整训练／验证曲线。
- `results/cnn_enhanced/trial_1/augmentation_examples.png`：入选增强方案的示意图。
- `results/scattering/scattering_feature_maps.png`：小波散射特征示意图。

每张 PNG 都有对应 PDF。汇总数值见 `results/comparison/summary.csv`。

已核验：三个 notebook 的全部代码单元执行成功；重新加载 8 个“方法设置 × trial”的模型，完整复现共 16,000 条保存的预测；49 组 PNG/PDF 图片成对存在；最终只存在两组 CNN 结果。检查记录见 `results/verification.json`。

## 实验设置

- 严格使用老师给出的两个 trial，每次训练／测试各 2,000 张。
- CNN 和 Scattering-SVM 在每个 trial 的训练集内部，用固定种子做分层 80/20 划分：1,600 张拟合、400 张验证。两个方法使用相同划分。
- 1-NN 无需调参，直接用 2,000 张训练图像拟合。
- CNN 最终只保留两组：普通几何增强，以及在此基础上采用入选增强组合的设置。网络、优化器、划分、训练预算和种子保持一致。
- 普通几何增强为旋转 ±10°、水平／垂直平移 ±2 像素、缩放 0.85–1.15。仿射变换只进行一次线性插值，背景填零，不翻转数字。
- 两组均最多训练 50 个 epoch，10 个 epoch 无验证 loss 改善时早停。通过验证 loss 选出 epoch 数后，重新初始化，在全部 2,000 张训练图像上训练选定的 epoch 数，再评估测试集。
- 额外增强仅依据本 trial 训练内部验证集选择。每个 trial 的方案独立冻结，不能汇总两个 trial 的验证成绩替任一 trial 选方案，因为它们的数据互为测试集。最终参数见 `cnn_config.json` 和各 trial 的 `config.json`。
- CNN 验证和测试时不做增强。弹性形变的强度表示位移向量的均方根长度（像素）；若入选方案包含形态学操作，则使用一次 3×3 十字形结构元素，随机选择腐蚀或膨胀，不叠加多次。
- 小波散射固定 `J=2, L=8, max_order=2`，得到 81 张 7×7 系数图，即 3,969 维特征。分类器使用 `SVC(kernel="linear")`（LIBSVM 实现），仅在 `C=[0.01, 0.1, 1.0]` 中选择，避免扩大实验范围。实现选择通过训练数据上的耗时检查决定。
- 固定散射特征不使用标签、不学习参数，可以缓存整个数据集的系数。**StandardScaler 和 SVM 只能在当前训练分区内拟合**；参数选定后才用当前 trial 的全部训练集重新拟合。
- 两个 trial 独立执行。不要依据已看到的测试分数调整参数、增强或网络后重新声称是独立测试结果。

所有配置保存在结果目录的 `config.json` 中。均值后的标准差采用两个 trial 的**样本标准差**（`ddof=1`），不是多次随机种子重复实验的标准差。一次种子的增强差异只能作为本次实验观察，不能据此声称统计显著。

## 图片和结果在哪里

所有图片同时保存为高分辨率 `.png` 和可缩放的 `.pdf`。notebook 也内嵌显示图片。

```text
results/
  data/                         数据样例、类别分布、像素分布、两个 trial 划分
  1nn/
    trial_1/                    模型、指标、预测、混淆矩阵、错误图片、每类召回率
    trial_2/
    trial_results.csv           两个 trial 的结果
    summary.csv                 平均准确率、标准差和耗时
  cnn_geometric/
    trial_1/
      history.csv               每个 epoch 的训练/验证 loss、准确率和累计时间
      learning_curves.png/pdf   用于选择 epoch 的训练/验证曲线
      validation_checkpoint.pt  仅在 1,600 张拟合集上训练的最佳验证模型
      final_model.pt            在全部 2,000 张训练图像上重新训练后的测试模型
      final_training_history.csv
      final_training_curves.png/pdf
    trial_2/
  cnn_enhanced/                  与上述结构相同，各 trial 包含入选增强方案的示意图
  scattering/
    trial_1/                    系数图、C 的验证结果及曲线、SVM 模型、评估图表
    trial_2/
  comparison/                   各方法成绩表、准确率比较图、CNN 增强收益图
  notebooks/                    已执行 notebook 的 HTML 版本
```

1-NN 和线性 SVM 没有 CNN 那种逐 epoch 的交叉熵训练过程，因此不编造 loss 曲线。1-NN 记录基线评估，Scattering-SVM 记录不同 C 的训练／验证准确率和耗时。CNN 的训练准确率在带增强且启用 dropout 时计算，验证准确率在无增强、关闭 dropout 时计算，两条曲线存在差异是正常的。

Scattering 的 `train_seconds` 是参数选择与 SVM 拟合时间，特征提取时间另存 `feature_extraction.json`。预测表的 `sample_index` 是图像在原始 4,000 张数据中的零基索引。不同方法的速度不能在忽略特征提取成本的情况下直接比较。

## 附加挑战

当前没有老师的挑战数据。三个 notebook 末尾给出了加载已保存模型的方法；拿到数据后应先按老师给出的方向、极性和尺寸规范整理成 `[N, 28, 28]`、取值 `[0,1]`、亮字黑底的数组，再预测。不能直接假设新 `.mat` 文件与原始图像排列一致。

本项目暂不擅自把两个 trial 的模型合并，也不使用额外 MNIST 数据；最终挑战训练方式按老师规则决定。挑战标签不能用于调参。

## 第三方引用

- 数据及评估协议：`PA-3-courseproject.pdf`。
- Bruna, J. and Mallat, S. *Invariant Scattering Convolution Networks*. https://arxiv.org/abs/1203.1513
- Andreux, M. et al. *Kymatio: Scattering Transforms in Python*. JMLR 21(60), 2020. https://jmlr.org/papers/v21/19-047.html
- OpenCV，形态学操作原理：https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html 。实际灰度腐蚀和膨胀由 SciPy 实现。
- Simard, P. Y., Steinkraus, D. and Platt, J. C. (2003). *Best Practices for Convolutional Neural Networks Applied to Visual Document Analysis*. https://www.microsoft.com/en-us/research/publication/best-practices-for-convolutional-neural-networks-applied-to-visual-document-analysis/
- 实现依赖 PyTorch、scikit-learn、SciPy、NumPy、pandas、Matplotlib 和 Jupyter；报告中应列出所用库。
