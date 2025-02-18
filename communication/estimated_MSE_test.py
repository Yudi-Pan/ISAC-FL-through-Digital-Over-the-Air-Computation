import MSE_estimate
import torch
import numpy as np
import matplotlib.pyplot as plt


# 定义参数范围和步长
Q_levels = np.arange(2, 21, 2)  # 从4到20，步长为2
num_users_values = np.arange(5, 21, 5)  # 从5到20，步长为5

# 固定其他参数
lambda_r = 1e-7
sigma2 = 1e-8  # 不同snr居然导致不同图像，神奇。
snr = 10 * np.log10(lambda_r / sigma2)

v_min = -1
v_max = 2
# 创建图像 - 不同 num_users 对应的 MSE
plt.figure(figsize=(10, 6))

# 循环遍历 num_users_values
for num_users in num_users_values:
    # 计算对应 num_users 下不同 Q_level 的 MSE
    mse_values = [MSE_estimate.calculate_MSE2(Q_level, v_min, v_max, num_users, np.sqrt(sigma2), lambda_r) for Q_level in Q_levels]

    # 绘制当前 num_users 对应的 MSE 曲线
    plt.plot(Q_levels, mse_values, label=f'num_users={num_users}')

# 添加标签和标题
plt.xlabel('Q_level')
plt.ylabel('MSE')
plt.title(f'e_MSE vs. Q_level, SNR={snr:.2f} dB')
plt.xticks(np.arange(min(Q_levels), max(Q_levels) + 1, 1))  # 设置 x 轴坐标为整数
plt.legend()
plt.grid(True)
# 保存图像
plt.savefig(f'e_MSE_vs_Q_level_with_SNR={snr:.2f}dB.png')  # 文件名可以自定义，后缀指定图片格式
plt.show()
