PyTorch实践：入门图像预处理与 CNN 实现
# 图像数据集处理
## 从这段代码开始

```python
import torch
import torchvision
import torchvision.transforms as transforms
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```

--- 

## 1. CIFAR-10 数据集加载
CIFAR-10 有 50,000 张训练图片；10,000 张测试图片
```python

trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)
```

- `root='./data'` → 数据存放路径  
- `train=True` → 加载训练集，FALSE为CIFAR-10的测试数据集
- `download=True` → 如果本地没有数据则自动下载  
- `transform=transform` → 定义图像预处理操作  

---

## 2. 数据预处理流程
数据在加载时会经历以下步骤：
1. **原始数据**：NumPy 数组，像素值范围 `0–255`  
2. **PIL.Image**：CIFAR-10 默认返回 PIL 格式图像  
3. **ToTensor()**：转换为 PyTorch 张量，形状 `[C,H,W]`，数值范围 `[0,1]` 
  - **C** = Channels（通道数，RGB=3）  
  - **H** = Height（高度）  
  - **W** = Width（宽度）   
4. **Normalize(mean,std)**：标准化到 `[-1,1]`，公式：
   \[
   x' = {x - mean}/{std}
   \]

- **mean** → 平均值，用于中心化  
- **std** → 标准差，用于缩放  
- 在标准化中手动指定，例如 `(0.5,0.5,0.5)`，也可以通过统计数据集真实分布来计算。  

### 数据流示例
#### 📸 假设原始数据
我们先定义一张 2×2 的小图片A，每个像素用 `[R,G,B]` 表示，范围是 `0~255`。

##### 图片 A（红色为主）
```
[[[255,   0,   0],   [128,   0,   0]],
 [[ 64,   0,   0],   [  0,   0,   0]]]
```


#### 🔄 流程转换

#####  1️⃣ 原始（NumPy数组）
- 数据类型：`uint8`
- 范围：`0~255`
- 形状：`(H, W, C)` → `(2, 2, 3)`

例子（图片 A）：
```
array([[[255,   0,   0],
        [128,   0,   0]],
       [[ 64,   0,   0],
        [  0,   0,   0]]], dtype=uint8)
```


#####  2️⃣ PIL.Image
- `torchvision.datasets.CIFAR10` 默认返回 **PIL.Image**。
- PIL 内部仍然是 `0~255` 的像素值，只是封装成了图像对象。
- 打印时通常显示：`<PIL.Image.Image image mode=RGB size=2x2>`


#####  3️⃣ Tensor（`transforms.ToTensor()`）
- 转换为 **PyTorch张量**。
- 数据类型：`float32`
- 范围：`[0.0, 1.0]`
- 形状：`(C, H, W)` → `(3, 2, 2)`（通道在前）

例子（图片 A）：
```
tensor([[[1.0000, 0.5020],
         [0.2510, 0.0000]],   # R 通道

        [[0.0000, 0.0000],
         [0.0000, 0.0000]],   # G 通道

        [[0.0000, 0.0000],
         [0.0000, 0.0000]]])  # B 通道
```
> 注意：255 → 1.0，128 → 0.502，64 → 0.251。


#####  4️⃣ 标准化（`transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))`）
公式：  
\[
x' = {x - mean}/{std}
\]
其中 `mean=0.5`，`std=0.5`。  
所以：  
- 输入 `[0,1]` → 输出 `[-1,1]`。

例子（图片 A，R 通道）：
```
原始值 1.000 → (1.000 - 0.5)/0.5 = +1.0
原始值 0.502 → (0.502 - 0.5)/0.5 ≈ +0.004
原始值 0.251 → (0.251 - 0.5)/0.5 ≈ -0.498
原始值 0.000 → (0.000 - 0.5)/0.5 = -1.0
```

最终张量（图片 A）：
```
tensor([[[ 1.0000,  0.0040],
         [-0.4980, -1.0000]],   # R 通道

        [[-1.0000, -1.0000],
         [-1.0000, -1.0000]],   # G 通道

        [[-1.0000, -1.0000],
         [-1.0000, -1.0000]]])  # B 通道
```

---

## 3. DataLoader 的使用
`DataLoader` 用于批量加载数据：
```python
import torch

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=4,
    shuffle=True,
    num_workers=2
)
```

- `batch_size=4` → 每批次 4 张图片  
- `shuffle=True` → 每个 epoch 打乱数据顺序，通常打乱训练数据集增加泛化能力，测试数据集不需打乱。
- `num_workers=2` → 使用两个子进程并行加载数据  

输出：
- `images` → `[batch_size, C, H, W]` 的张量  
- `labels` → `[batch_size]` 的类别索引  


---

## 4. 图像与标签对应关系
- 数据集始终以 `(image, label)` 二元组形式保存  
- `images` → 经历预处理流程  
- `labels` → 整数类别索引，不参与图像预处理  


---

## 5. 可视化工具
使用 `make_grid` 拼接 batch 图像：
```python
import torchvision.utils as vutils
import matplotlib.pyplot as plt

dataiter = iter(trainloader)
images, labels = next(dataiter)

grid = vutils.make_grid(images, nrow=4, normalize=True)
plt.imshow(grid.permute(1, 2, 0))  # 转换为 HWC 格式
plt.show()
```

---


# CNN 实现

## 从这段代码开始

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 第一层卷积：输入通道3（RGB），输出通道6（或者说是6个卷积核），卷积核大小5x5
        self.conv1 = nn.Conv2d(3, 6, 5)
        # 最大池化层：2x2窗口，步长2，效果是尺寸缩小一半
        self.pool = nn.MaxPool2d(2, 2)
        # 第二层卷积：输入通道6，输出通道16，卷积核大小5x5
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 全连接层1：输入特征 16*5*5，输出120
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 全连接层2：输入120，输出84
        self.fc2 = nn.Linear(120, 84)
        # 全连接层3：输入84，输出10（对应10个类别）
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 卷积1 + ReLU + 池化
        x = self.pool(F.relu(self.conv1(x)))
        # 卷积2 + ReLU + 池化
        x = self.pool(F.relu(self.conv2(x)))
        # 展平：保持 batch 维度不变，把剩下的特征拉直
        x = torch.flatten(x, 1)
        # 全连接层1 + ReLU
        x = F.relu(self.fc1(x))
        # 全连接层2 + ReLU
        x = F.relu(self.fc2(x))
        # 全连接层3（输出层，不加激活）
        x = self.fc3(x)
        return x
```


### 🔹 网络结构流程
  **卷积 → ReLU → 池化 → 卷积 → ReLU → 池化 → 展平 → 全连接 → 全连接 → 输出层**  
1. **输入**：比如一张 RGB 图片，形状 `[batch_size, 3, 32, 32]`。  
2. **conv1**：卷积核大小 5×5，输出通道 6 → 得到 `[batch_size, 6, 28, 28]`。  
3. **pool**：2×2 最大池化 → `[batch_size, 6, 14, 14]`。  
4. **conv2**：卷积核大小 5×5，输出通道 16 → `[batch_size, 16, 10, 10]`。  
5. **pool**：2×2 最大池化 → `[batch_size, 16, 5, 5]`。  
6. **flatten**：展平为 `[batch_size, 16*5*5] = [batch_size, 400]`。  
7. **fc1**：线性层 → `[batch_size, 120]`。  
8. **fc2**：线性层 → `[batch_size, 84]`。  
9. **fc3**：线性层 → `[batch_size, 10]`（对应 10 个分类结果）。  

--- 

## torch.nn.Conv2d 卷积
Conv → Convolution（卷积）
2d → 2-dimensional（二维）

在 **PyTorch 的 `nn.Conv2d(in_channels, out_channels, kernel_size)`** 中：  

- **`in_channels=3`** → 输入有 3 个通道（RGB）。  
- **`out_channels=6`** → 输出有 6 个通道，这就意味着这一层会学习 **6 个不同的卷积核（filter）**。  
- **`kernel_size=5`** → 每个卷积核的大小是 5×5。  


### 🔹 更具体地说
- 每个卷积核的形状是 `[in_channels, kernel_size, kernel_size]` → `[3, 5, 5]`。  
- 因为有 6 个卷积核，所以这一层的权重张量形状是：
  ```
  [out_channels, in_channels, kernel_size, kernel_size]
  = [6, 3, 5, 5]
  ```
- 每个卷积核会在输入图像上滑动，提取不同的特征（比如边缘、纹理、颜色组合等）。  
- 最终输出的特征图（feature map）有 6 个通道，每个通道对应一个卷积核的结果。
- 卷积核与输入图像每一步的计算方式：
  ```
  ①对每个通道分别做“逐元素相乘求和”。之后没有加偏置这一步，是因为卷积核初始化是没有偏执为0。
  ②然后把 3 个通道的结果再相加，得到最终输出通道的一个像素值
  ```
- 卷积核的初始化：
  ```
  PyTorch 的 Conv2d 默认使用：Kaiming Uniform：从一个均匀分布 U(-bound, bound) 中随机采样。
  bound = sqrt(6 / in_channels × kernel_size × kernel_size)
      = sqrt(6 / 3 × 5 × 5) ≈ 0.2828
  所以每个卷积核元素都从 [-0.28, 0.28] 之间随机采样
  
  ```

---

## 池化层 (Pooling Layers)
- **`nn.MaxPool2d(kernel_size=2, stride=2)`**  
  - 每次取 2×2 区域的最大值  
  - 步长为 2 → 特征图尺寸缩小一半  
- 作用：下采样，减少参数量，保留显著特征。

**常见池化方式对比表**

| 池化方式 | 计算原理 | 适用场景 |
|----------|----------|----------|
| **Max Pooling (最大池化)** | 在池化窗口内取最大值 | 保留最显著特征，常用于图像识别和分类任务 |
| **Average Pooling (平均池化)** | 在池化窗口内取平均值 | 平滑特征图，减少噪声，常用于特征压缩 |
| **Global Average Pooling (全局平均池化)** | 对整个特征图取平均值，输出为每个通道一个值 | 替代全连接层，减少参数，常用于分类网络（如 GoogLeNet, ResNet） |
| **Global Max Pooling (全局最大池化)** | 对整个特征图取最大值，输出为每个通道一个值 | 保留最强响应，常用于检测任务或注意力机制 |
| **Stochastic Pooling (随机池化)** | 在池化窗口内按概率随机选择一个值（概率与激活值大小相关） | 增加随机性，防止过拟合，常用于数据增强 |
| **Mixed Pooling (混合池化)** | 在同一层中结合最大池化和平均池化（如随机选择或加权组合） | 提取多样化特征，提升模型鲁棒性 |
| **L2 Pooling (平方根平均池化)** | 在池化窗口内计算平方和再开方：\(\sqrt{\sum x^2}\) | 保留能量信息，常用于图像检索和特征匹配 |

---

## torch.flatten 展平

```python
torch.flatten(input, start_dim=1, end_dim=-1)
```
- **作用**：把张量在指定维度范围内展平（拉直成一维）。  
- `start_dim=1` → 从第 1 个维度开始展平（保留第 0 维，也就是 batch 维度）。  
- `end_dim=-1` → 默认到最后一个维度结束。  


### 🔹 举例：CNN 中的典型情况
假设输入 `x` 是卷积层输出后的特征图，形状：
```
[batch_size, channels, height, width]
```

例如：
```
x.shape = [4, 16, 5, 5]
```
- 4 → batch_size  
- 16 → 通道数  
- 5, 5 → 特征图的高和宽  


### 🔹 执行 `torch.flatten(x, 1)`
- 保留第 0 维（batch_size = 4）  
- 从第 1 维开始展平，把 `[16, 5, 5]` 拉直成一个向量  

结果：
```
x.shape = [4, 400]
```
因为：
\[
16 * 5 * 5 = 400
\]

---

### nn.Linear全链接层
把高维特征压缩到更小的空间，同时通过权重学习提取更抽象的模式。  
  `fc1 = nn.Linear(400, 120)`
- 输入：400 维特征  
- 输出：120 维特征  
- <b>权重矩阵形状</b>：`[120, 400]`，每个输出维度都是输入 400 维的加权组合。
<br/>
每一层都是一个 **线性变换 + 激活函数 (ReLU)**，逐步压缩特征维度。  
三次压缩后的 10 维向量就是分类的 **logits**，通常会再经过 `Softmax` 转换成概率分布。


# 参考
[Training a Classifier](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#define-a-convolutional-neural-network)
[CNN原理](https://www.bilibili.com/video/BV1eh1LBjEkx/?share_source=copy_web&vd_source=2942ada437a134a36e1c9049cdf770dd)
[学习记录](https://copilot.microsoft.com/shares/M5wT7psfzfTu1qyRWDxgH)


```
最后来一首四言诗

《张量颂》
维度如川，层层相叠； 元素如星，点点相接。 方格定影，万象皆收； 数理之美，天地同流。
坐标织网，秩序如经； 数据成诗，抽象有情。 简而不凡，深而无穷； 张量之境，万物归融。

```
