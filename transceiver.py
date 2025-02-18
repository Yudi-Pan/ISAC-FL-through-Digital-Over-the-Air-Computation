import numpy as np
import torch

# def phi(x, gamma_r):
#     # 计算 x 属于哪个量化区间 ell
#     ell = np.floor(x * gamma_r).astype(int)
#
#     # 根据概率生成随机值决定是返回 ell 还是 ell + 1
#     decision = np.random.rand(x.shape) < (x * gamma_r - ell)
#     return np.where(decision, np.minimum(ell + 1, gamma_r), ell)
def phi(x, gamma_r):
    # 使用广播将 gamma_r 扩展为与 x 相同的形状
    gamma_r = gamma_r * torch.ones_like(x)
    # 计算 x 属于哪个量化区间 ell
    ell = (x * gamma_r).floor().long()

    # 根据概率生成随机值决定是返回 ell 还是 ell + 1
    decision = torch.rand_like(x) < (x * gamma_r - ell).float()
    result = torch.where(decision, torch.min(ell + 1, gamma_r), ell)

    return result

def QAM(data, Q_level, v_min, v_max):
    # 标准化
    x = (data - v_min) / (v_max - v_min)

    # 量化
    x_q = phi(x, Q_level)

    # 计算量化后的值
    # v_q = v_min + (v_max - v_min) / gamma_r * x_q

    # 检查数组长度，如果为奇数，添加一个0
    if x_q.size(0) % 2 != 0:
        x_q = torch.cat((x_q, torch.tensor([0], dtype=torch.float32).cuda()))
        # flag = True

    # 将数组的奇数位元素作为实部，偶数位元素作为虚部
    real_parts = (2 / Q_level) * x_q[0::2] - 1
    imag_parts = (2 / Q_level) * x_q[1::2] - 1
    complex_arr = torch.view_as_complex(torch.stack((real_parts, imag_parts), dim=1))  # 组合成复数数组

    return complex_arr


def channel(complex_arr, lambda_r, sigma2):
    """
    complex_arr: aggregate Modulated symbols for K devices, size of (d, 1)
    lambda_r: Transmit power scaling factors for devices
    sigma2: Variance of the additive white Gaussian noise
    """
    d = complex_arr.size(0)  # Dimension of the noise vector and modulated symbols

    # 将 lambda_r 和 sigma2 转换为 PyTorch 张量
    lambda_tensor = torch.tensor(lambda_r).cuda()
    sigma2_tensor = torch.tensor(sigma2).cuda()

    # 计算 y
    y = torch.sqrt(lambda_tensor) * complex_arr

    # 添加高斯白噪声
    n_r_real = torch.normal(0, torch.sqrt(sigma2_tensor / 2), size=(d,)).cuda()
    n_r_imag = torch.normal(0, torch.sqrt(sigma2_tensor / 2), size=(d,)).cuda()
    n_r = torch.view_as_complex(torch.stack((n_r_real, n_r_imag), dim=1))
    y += n_r

    return y


def receiver(y, Q_level, K, lambda_r, v_min, v_max, flag=False):
    # 生成实数和虚数部分的可能值
    values = torch.linspace(-K, K, steps=int(Q_level * K + 1)).cuda()

    # 使用 meshgrid 生成两两组合的实数和虚数部分
    real_part, imag_part = torch.meshgrid(values, values)

    # 将实数和虚数部分组合成复数星座点
    constellation = (real_part + 1j * imag_part).flatten()

    # 切割 y 成小的子张量，每个子张量大小为 batch_size
    batch_size = 50000  # 可以根据显存大小和计算性能调整
    num_batches = (len(y) + batch_size - 1) // batch_size  # 向上取整
    batches = [y[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]

    # 初始化空张量用于存储最近的星座点
    sym_m_list = []

    # 计算每个子张量中的 x 值与最近的星座点的距离，并找到最近的星座点
    for batch in batches:
        distances = torch.abs((batch / torch.sqrt(lambda_r)).unsqueeze(1) - constellation)
        min_values, min_indices = torch.min(distances, dim=1)
        closest_values = torch.index_select(constellation, 0, min_indices)
        sym_m_list.append(closest_values)

    # 将每个子张量计算得到的最近星座点组合成一个张量
    sym_m = torch.cat(sym_m_list)

    # 获取复数数组的实部和虚部
    real_parts = sym_m.real
    imag_parts = sym_m.imag

    # 创建一个新数组，长度是原数组的两倍
    real_arr = torch.empty(real_parts.size(0) + imag_parts.size(0), dtype=real_parts.dtype).cuda()

    # 实部和虚部交替排列
    real_arr[0::2] = real_parts
    real_arr[1::2] = imag_parts

    # 如果flag为True，则删除最后一个元素
    if flag:
        real_arr = real_arr[:-1]
    # sum_x_q = (Q_level/2) * (real_arr + K)
    # sum_v_q = K * v_min + (v_max - v_min) / Q_level * sum_x_q
    v_tilde = K * v_min + (v_max - v_min) / 2 * (real_arr + K)
    # 在函数结束时手动释放不再需要的显存
    torch.cuda.empty_cache()

    return v_tilde


def MSE_quantiz(Q_level, num_users):
    # 假设 signal_send 是从 v_min 到 v_max 均匀分布的实数信号
    signal_dim = 10000  # 信号维度
    v_min = -1
    v_max = 2
    # 生成信号，假设为均匀分布
    x = torch.rand(num_users, signal_dim).cuda()
    v_q = torch.zeros_like(x).cuda()
    for idx in range(num_users):
        # 量化
        v_q[idx, :] = v_min + (v_max - v_min) / Q_level * phi(x[idx, :], Q_level)

    # 按行求和
    x_sum = torch.sum(v_min + (v_max - v_min) * x, dim=0)
    x_q_sum = torch.sum(v_q, dim=0)

    # 计算差异
    diff = x_q_sum - x_sum

    # 计算 MSE
    mse = torch.mean(diff ** 2).item()

    return mse


def generate_fading(large_scale_fading, threshold=0.05):
    """
    生成小尺度衰落，自动重试直到所有值超过阈值

    Args:
        large_scale_fading: 大尺度衰落系数 (K,)
        threshold: 最小幅度阈值 (默认0.1)

    Returns:
        组合后的信道状态 (K, 1)
    """
    K = len(large_scale_fading)

    while True:
        # 生成瑞利衰落
        h = (np.random.normal(0, 1, (K, 1)) +
             1j * np.random.normal(0, 1, (K, 1))) / np.sqrt(2)

        # 检查所有元素的幅度是否达标
        if np.all(np.abs(h) >= threshold):
            return large_scale_fading[:, None] * h
