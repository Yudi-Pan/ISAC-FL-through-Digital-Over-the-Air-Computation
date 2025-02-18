from scipy.special import erfc
import numpy as np
import torch
from torch.distributions import Normal

def cal_probability(m, K, gamma_r):
    """
    The reason for using logarithms is numerical stability problem.
    :param m:
    :param K:
    :param gamma_r:
    :return:
    """
    summation = 0
    for p in range(int(m / (gamma_r + 1)) + 1):
        numerator = np.sum(np.log(np.arange(m - p * (gamma_r + 1) + 2, m - p * (gamma_r + 1) + K + 1)))
        denominator = np.sum(np.log(np.arange(1, p + 1))) + np.sum(np.log(np.arange(1, K - p + 1)))
        summation += (-1) ** p * np.exp(numerator - denominator)

    probability = K / (gamma_r + 1) ** K * summation
    return probability


def calculate_Expectation(m, gamma_r, v_min, v_max, K, sigma, lambda_r):
    alpha = (v_max - v_min) / 2

    summation1 = 0
    if m != 0:
        for ell in range(1, m + 1):
            arg = (2 * ell - 1) * np.sqrt(lambda_r) / (np.sqrt(2) * gamma_r * sigma)
            summation1 += (2 * ell - 1) * 0.5 * erfc(arg)
            # summation1 += (2 * ell - 1) * 0.5 * erfc((2 * ell - 1) * np.sqrt(lambda_r) / (np.sqrt(2) * gamma_r *
            # sigma))

    summation2 = 0
    if m != gamma_r * K:
        for ell in range(1, int(gamma_r * K - m) + 1):
            arg = (2 * ell - 1) * np.sqrt(lambda_r) / (np.sqrt(2) * gamma_r * sigma)
            summation2 += (2 * ell - 1) * 0.5 * erfc(arg)
            # summation2 += (2 * ell - 1) * 0.5 * erfc((2 * ell - 1) * np.sqrt(lambda_r) / (np.sqrt(2) * gamma_r *
            # sigma))

    result = 4 * alpha ** 2 * (summation1 + summation2) / gamma_r ** 2
    return result


def calculate_MSE2(gamma_r, v_min, v_max, K, sigma, lambda_r):
    result = 0
    for m in range(int(gamma_r * K) + 1):
        pro = cal_probability(m, K, gamma_r)
        exp = calculate_Expectation(m, gamma_r, v_min, v_max, K, sigma, lambda_r)
        result += pro * exp
    return result


# def calculate_Ed(m, gamma_r, v_min, v_max, K, sigma_n, lambdar):
#     # 导数
#
#     # 假设gamma_r、sigma_n和lambdar是已定义的张量
#     assert lambdar > 0
#     # 将所有的张量都移到GPU上
#     gamma_r = torch.tensor(gamma_r).cuda()
#     sigma_n = torch.tensor(sigma_n).cuda()
#     lambdar = torch.tensor(lambdar).cuda()
#     alpha = ((v_max - v_min) / 2).cuda()
#     summation1 = torch.tensor([0.0]).cuda()
#     summation2 = torch.tensor([0.0]).cuda()
#
#     # 定义正态分布
#     normal_dist = Normal(0, 1)
#
#     # 检查m是否为0
#     if m != 0:
#         for ell in range(1, m + 1):
#             arg1 = torch.tensor((2 * ell - 1) ** 2).cuda() / (2 * gamma_r * sigma_n * torch.sqrt(lambdar))
#             summation1 += arg1 * normal_dist.cdf((2 * ell - 1) * torch.sqrt(lambdar) / (gamma_r * sigma_n))
#
#     # 检查m是否等于gamma_r * K
#     if m != gamma_r * K:
#         for ell in range(1, int(gamma_r * K - m) + 1):
#             arg1 = torch.tensor((2 * ell - 1) ** 2).cuda() / (2 * gamma_r * sigma_n * torch.sqrt(lambdar))
#             summation2 += arg1 * normal_dist.cdf((2 * ell - 1) * torch.sqrt(lambdar) / (gamma_r * sigma_n))
#
#     result = -4 * alpha ** 2 * (summation1 + summation2) / gamma_r ** 2
#
#     return result
#
#
# def calculate_D(lambdar, V, B, gamma_r, K, v_min, v_max, sigma2):
#     result = 0
#     for m in range(int(gamma_r * K + 1)):
#         pro = cal_probability(m, K, gamma_r)
#         Expe = calculate_Ed(m, gamma_r, v_min, v_max, K, torch.sqrt(torch.tensor(sigma2)), lambdar)
#         result += pro.cpu() * Expe.cpu()
#
#     result = V * result.cpu() + B
#     return result
