import transceiver as trans
import torch
import numpy as np
import matplotlib.pyplot as plt


def realMSE_comm_align(Q_level, lambda_r, num_users, sigma2):
    # 假设 signal_send 是从 v_min 到 v_max 均匀分布的实数信号
    signal_dim = 10000  # 信号维度
    v_min = -1
    v_max = 2

    # 生成信号，假设为均匀分布
    x = torch.rand(num_users, signal_dim).cuda()
    # signal_send = x * (v_max - v_min) + v_min

    # 将 1 维张量 reshape 成 2 维张量
    cum_signal = torch.view_as_complex(torch.zeros([signal_dim // 2, 2])).cuda()
    x_q = torch.zeros_like(x).cuda()
    for idx in range(num_users):
        # 量化
        x_q[idx, :] = trans.phi(x[idx, :], Q_level)
        # 将数组的奇数位元素作为实部，偶数位元素作为虚部
        real_parts = (2 / Q_level) * x_q[idx, 0::2] - 1
        imag_parts = (2 / Q_level) * x_q[idx, 1::2] - 1
        cum_signal += torch.view_as_complex(torch.stack((real_parts, imag_parts), dim=1))  # 组合成复数数组
    cum_signal = trans.channel(cum_signal, lambda_r, sigma2)
    v_tilde = trans.receiver(cum_signal, Q_level, num_users,
                             torch.tensor(lambda_r).cuda(),
                             v_min, v_max, False)

    # 计算聚合信号的 Ground Truth，即按行求平均
    ground_truth = x_q.sum(dim=0)
    ground_truth = num_users * v_min + (v_max - v_min) / Q_level * ground_truth
    # 计算 v_tilde 和 ground_truth 之间的差值
    diff = v_tilde - ground_truth

    # 计算差值的平方
    diff_squared = diff ** 2

    # 计算 MSE
    mse = diff_squared.mean().item()
    return mse


# 定义参数范围和步长
Q_levels = np.arange(2, 65, 2)  # 从4到20，步长为2
num_users_values = np.arange(5, 11, 5)  # 从5到20，步长为5

# 固定其他参数
lambda_r = 1e-6
sigma2 = 1e-9  # 不同snr居然导致不同图像，神奇。
snr = 10 * np.log10(lambda_r / sigma2)
# # 创建图像 - 不同 Q_level 对应的 MSE
# plt.figure(figsize=(10, 6))
#
# # 循环遍历 Q_levels
# for Q_level in Q_levels:
#     # 计算对应 Q_level 下不同 num_users 的 MSE
#     mse_values = [realMSE_comm_align(Q_level, lambda_r, num_users, sigma2) for num_users in num_users_values]
#
#     # 绘制当前 Q_level 对应的 MSE 曲线
#     plt.plot(num_users_values, mse_values, label=f'Q_level={Q_level}')
#
# # 添加标签和标题
# plt.xlabel('num_users')
# plt.ylabel('MSE')
# plt.title('MSE vs. num_users for different Q_level')
# plt.xticks(np.arange(min(Q_levels), max(Q_levels)+1, 1))  # 设置 x 轴坐标为整数
# plt.legend()
# plt.grid(True)
# plt.show()

# 创建图像 - 不同 num_users 对应的 MSE
plt.figure(figsize=(10, 6))

# 循环遍历 num_users_values
for num_users in num_users_values:
    # 计算对应 num_users 下不同 Q_level 的 MSE
    mse_values = [realMSE_comm_align(Q_level, lambda_r, num_users, sigma2) for Q_level in Q_levels]

    # 绘制当前 num_users 对应的 MSE 曲线
    plt.plot(Q_levels, mse_values, label=f'num_users={num_users}')

# 添加标签和标题
plt.xlabel('Q_level')
plt.ylabel('MSE')
plt.title(f'MSE vs. Q_level, SNR={snr:.2f} dB')
plt.xticks(np.arange(min(Q_levels), max(Q_levels) + 1, 1))  # 设置 x 轴坐标为整数
plt.legend()
plt.grid(True)
# 保存图像
plt.savefig(f'MSE_vs_Q_level_with_SNR={snr:.2f}dB.png')  # 文件名可以自定义，后缀指定图片格式
plt.show()
