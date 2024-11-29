import torch
import numpy as np
import matplotlib.pyplot as plt
import torch

# 1. 加载 .pt 文件
file_path = "../../ReluLLaMA-13B-PowerInfer-GGUF/activation/activation_2.pt"  # 替换为你的文件路径
data = torch.load(file_path)


data, _ = torch.sort(data, descending=True)


# if isinstance(data, torch.Tensor) and data.ndim == 1:
#     # 转换为 NumPy 数组以便统计
#     data_numpy = data.numpy()

#     # 统计值在不同范围内的个数
#     above_200k = (data_numpy > 200000).sum()
#     between_100k_200k = ((data_numpy >= 100000) & (data_numpy <= 200000)).sum()
#     between_50k_100k = ((data_numpy>=50000)&(data_numpy<=100000)).sum()
#     below_50k = (data_numpy < 50000).sum()

#     # 打印统计结果
#     print(f"值在200000以上的有: {above_200k} 个")
#     print(f"值在100000-200000之间的有: {between_100k_200k} 个")
#     print(f"值在50000-100000之间的有: {between_50k_100k} 个")
#     print(f"值在50000以下的有: {below_50k} 个")
# else:
#     print("The loaded data is not a 1D vector. Please check the file content.")







# data = data * -1.0
# data = data.view(-1, 256)
# data = data.sum(dim=1)
# data = data.tolist()


# 2. 检查数据类型
# if isinstance(data, torch.Tensor):
#     print(f"Loaded data is a tensor with shape: {data.shape}")
#     print(f"Data type: {data.dtype}")
    
#     # 3. 分析向量大小
#     if data.ndimension() == 1:
#         print(f"The vector size is: {data.size(0)}")
#     else:
#         print(f"The tensor is multi-dimensional with shape: {data.shape}")

# elif isinstance(data, dict):
#     print("Loaded data is a dictionary. Keys:")
#     for key, value in data.items():
#         if isinstance(value, torch.Tensor):
#             print(f"Key: {key}, Tensor shape: {value.shape}, Data type: {value.dtype}")
#         else:
#             print(f"Key: {key}, Value type: {type(value)}")

# elif isinstance(data, list):
#     print(f"Loaded data is a list with {len(data)} items.")
#     for i, item in enumerate(data[:5]):  # 示例打印前5个
#         if isinstance(item, torch.Tensor):
#             print(f"Item {i}: Tensor shape: {item.shape}, Data type: {item.dtype}")
#         else:
#             print(f"Item {i}: Type: {type(item)}")
# else:
#     print(f"Loaded data is of type: {type(data)}")


# print("First few elements of the tensor (if applicable):")
# print(data[:5] if isinstance(data, torch.Tensor) else "Not a tensor")



if isinstance(data, torch.Tensor) and data.ndim == 1:
    # 绘制折线图
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(data)), data.numpy(), label='Activation Data', linewidth=1)
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("Activation Data Trend")
    plt.grid(alpha=0.5)
    plt.legend()
    
        # 保存图像到文件
    output_file = "13B_activation_sort_2.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')  # 300 DPI 高分辨率
    print(f"图像已保存到文件: {output_file}")
else:
    print("The loaded data is not a 1D vector. Please check the file content.")