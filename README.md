# MNIST 手写数字识别（C++ 实现）

一个基于 **LibTorch** 和 **OpenCV** 的入门级人工智能程序，使用两层全连接神经网络对 MNIST 手写数字进行训练和预测，并可视化预测结果。

---

## 功能特性

- 使用 LibTorch（PyTorch 的 C++ 前端）构建和训练模型
- 简单的两层全连接神经网络（784 → 500 → 10），带 Kaiming 初始化
- 自动下载 MNIST 数据集（包含 60,000 个训练样本和 10,000 个测试样本）
- Adam 优化器快速收敛，测试准确率可达 **96% 以上**
- 使用 OpenCV 将样本图像保存为 BMP 文件，并自动调用系统图片查看器
- 训练前可视化 5 个训练样本
- 训练后可视化 20 个测试样本的预测结果（真实标签 vs 预测标签）
- 完全使用 C++ 编写，适合学习深度学习底层实现和 LibTorch 应用

---

## 环境要求

| 名称          | 版本/说明                                |
| ------------- | ---------------------------------------- |
| 操作系统      | Windows 10/11（64 位）                  |
| 编译器        | Visual Studio 2022（或 2019，需支持 C++17） |
| CMake         | 3.21 或更高版本（用于自动部署 DLL）       |
| LibTorch      | PyTorch 1.8+ 的 Windows CPU 发行版      |
| OpenCV        | 4.x（通过 vcpkg 安装）                   |
| vcpkg         | C++ 包管理器（用于安装 OpenCV）          |

> 如果需要 `GPU` 训练，请下载 `LibTorch` 的 `CUDA` 版本，并确保安装了 `CUDA Toolkit`。

---

## 项目结构

项目根目录/
| :------------------------------------------------|
 ├── CMakeLists.txt # CMake 构建配置文件           |
 ├── mnist.cpp # 主程序源代码                      |
 ├── data/ # MNIST 数据集（程序自动下载）           |
 │ └── MNIST/                                     |
 │ ├── train-images-idx3-ubyte                    |
 │ ├── train-labels-idx1-ubyte                    |   
 │ ├── t10k-images-idx3-ubyte                     |
 │ └── t10k-labels-idx1-ubyte                     |
 ├── mnist.exe # 编译生成的可执行文件（在根目录）    |
 └── *.dll # 运行时所需的动态链接库（自动复制）      |

---
训练完成后，根目录会生成以下图像文件：
- `train_sample_0.bmp` ~ `train_sample_4.bmp`
- `test_sample_0.bmp` ~ `test_sample_19.bmp`

---

## 环境配置

### 1. 安装 LibTorch

1. 访问 [PyTorch 官网](https://pytorch.org) 下载 **LibTorch 的 Windows 发行版**（选择 `CPU` 或 `CUDA`，推荐 `CPU` 版本以简化配置）。
   
2. 将下载的压缩包解压到固定路径，例如 `D:/libtorch`。
   
3. 确保解压后的文件夹包含 `lib`、`include`、`share` 等子目录。
   
---
### 2. 安装 vcpkg 和 OpenCV

1. 克隆 vcpkg 仓库并引导安装：
   ```powershell
   git clone https://github.com/microsoft/vcpkg.git
   cd vcpkg
   .\bootstrap-vcpkg.bat
   ```
2. 使用 vcpkg 安装精简版 OpenCV（仅包含核心、图像处理和图像保存模块）：
   ```powershell
   .\vcpkg install opencv4[core,imgproc,imgcodecs]:x64-windows
   ```
  说明：该项目仅使用 `OpenCV` 保存 BMP 图像，不需要`GUI` 功能（如 `highgui` 模块），因此精简安装即可。

3. 记录 `vcpkg` 的安装路径（例如 `D:/vcpkg`），后续 `CMake` 配置会用到。
---   
### 3. 下载项目源码
将 `CMakeLists.txt` 和 `mnist.cpp` 放入同一个项目目录，例如 `D:/DNN_v2_0/DNN`。

---
## 编译和运行
### 1. 生成 CMake 项目
   打开 `PowerShell` 并进入项目根目录：
```
   cd D:/DNN_v2_0/DNN
   mkdir build
   cd build
   cmake .. -DCMAKE_TOOLCHAIN_FILE="D:/vcpkg/scripts/buildsystems/vcpkg.cmake"
```
   请将 `CMAKE_TOOLCHAIN_FILE` 的路径改为你的实际 `vcpkg` 路径。

###   2. 编译 Release 版本
```
   cmake --build . --config Release
```
编译成功后，可执行文件 `mnist.exe` 和依赖的 DLL 会自动部署到项目根目录。

### 3. 运行程序
回到项目根目录并运行：
```powershell
   cd ..   
   .\mnist.exe
```
---


## 使用说明
1. 首次运行：程序将自动从 `MNIST` 官网下载数据集并保存在 `./data/MNIST/`目录（或`./data`目录）下。请确保网络通畅。
2. 训练过程：控制台会输出每个 `epoch` 的平均损失值，10 个 `epoch` 后测试准确率将显示。
3. 图像输出：
   ◦ 训练开始前会保存前 5 个训练样本的 BMP 图像，并用系统默认图片查看器打开。
   ◦ 训练结束后会保存 20 个测试样本的 BMP 图像，图上标注了真实标签（绿色）和模型预测标签（红色）。
---
## 自定义训练参数
   可以在 `mnist.cpp` 中找到以下变量并按需修改：
```
   const int64_t input_size = 784;   // 输入特征数（28*28）
   const int64_t hidden_size = 500;  // 隐藏层神经元数量
   const int64_t num_classes = 10;   // 输出类别数
   const int64_t batch_size = 100;   // 训练批大小
   const double lr = 0.001;          // 学习率
   const int num_epochs = 10;        // 训练轮数
```   
   修改后重新编译即可生效。
   
---
## 常见问题
### Q1: 运行时提示“找不到 *.dll”
原因：缺少 `LibTorch` 或 `OpenCV` 的运行时动态链接库。
解决：
   • 确保编译时使用了 `CMake 3.21+` 并启用了自动 DLL 复制功能（已在 `CMakeLists.txt` 中配置）。
   • 如仍缺失，可手动将 `D:/libtorch/lib/*.dll` 和 `D:/vcpkg/installed/x64-windows/bin/*.dll`复制到 `mnist.exe` 所在目录。
### Q2: 提示“Error opening images file at ./data/...”
原因：MNIST 数据文件下载失败或路径不正确。
解决：
   • 检查 `data/MNIST/`(或`data`) 目录下是否存在 4 个 `ubyte`文件，且大小正常。
   删除 data 文件夹让程序重新下载。
   • 若下载失败，可手动从 MNIST 官网 下载 `.gz` 文件，解压后放入 `data/MNIST/`，文件名必须与上述完全一致（区分大小写）。
### Q3: OpenCV 报错“could not find a writer for the specified extension”
原因：精简版 OpenCV 可能缺少对应图像格式的编码器。
解决：
   • 代码中已使用 `.bmp` 格式，BMP 为 `OpenCV` 内置支持，无需额外编码器。
   • 如果仍报错，请检查是否在安装 `OpenCV` 时正确包含了 `imgcodecs`特性。
### Q4: 测试准确率较低（< 90%）
可能原因及解决方法：
   训练不充分：增加 `num_epochs` 至 15~20。
   学习率不合适：尝试将 `lr` 改为 0.0005 或 0.0001。
   优化器选择：可将 `Adam` 替换为带动量的 `SGD`：
```         
   torch::optim::SGD optimizer(net.parameters(), torch::optim::SGDOptions(lr).momentum(0.9));
```
---
## 代码说明
### 模型定义
`NeuralNet` 类继承自 `torch::nn::Module`，包含两个线性层，并在构造时使用 `Kaiming` 初始化（与 `PyTorch` 默认行为一致）。
### 数据预处理
   • `MNIST` 原始图像为 [1, 28, 28] 的张量（值范围 [0, 1]）。
   • 标准化：(img - 0.1307) / 0.3081，使其均值为 0、标准差为 1。
   • 展平为 [1, 784] 输入到全连接网络。
### 训练循环
   • 每个 `epoch` 前随机打乱训练集索引。
   • 手动构建 `batch`，通过 `torch::cat` 拼接张量。
   • 使用 `CrossEntropyLoss` 和 `Adam` 优化器更新权重。
### 可视化
   • `tensor_to_mat` 函数将张量反标准化并转换为 `OpenCV` 的 `cv::Mat`，放大到 280x280 像素。
   • 使用 `cv::putText` 添加标签文字。
   • 通过 `cv::imwrite` 保存为 BMP 文件，并用 `system("start ...") `调用系统默认程序打开。

---

## 许可证
   本项目仅用于学习和研究目的，引用时须注明来源。使用的 `MNIST` 数据集遵循其原始许可。`LibTorch` 和 `OpenCV` 的许可证分别遵循 `PyTorch` 和 `OpenCV` 的相关条款。
      
   > **重要（！！！！！！！！！）： 引用或采用本项目代码或思路、想法须自担风险，本项目不承担代码使用产生的任何风险和后果**。
---
## 联系与反馈

如有问题或建议，欢迎通过项目仓库的 Issue 功能提出。

---
文档生成日期：2026年6月26日.
