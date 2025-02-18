import numpy
import torch

import transceiver as trans


def aggregate_D_Air(signal_send, v_min, v_max, K, Q_level, lambda_r):
    cum_signal = torch.cat([(torch.zeros_like(grad)).flatten() for grad in signal_send[0]])
    # 检查数组长度，如果为奇数，添加一个0
    if cum_signal.size(0) % 2 != 0:
        cum_signal = torch.cat((cum_signal, torch.tensor([0], dtype=torch.float32).cuda()))
        flag = True
    else:
        flag = False
    # 将 1 维张量 reshape 成 2 维张量
    cum_signal = torch.view_as_complex(cum_signal.view(cum_signal.size(0) // 2, 2))

    for m_device in range(K):
        # 压缩成一维向量
        flat_grad = torch.cat([tensor.flatten() for tensor in signal_send[m_device]])
        # 对一维向量进行某些处理
        cum_signal += trans.QAM(flat_grad, Q_level, v_min, v_max)  # all k' flag are the same
    cum_signal = trans.channel(cum_signal, lambda_r, 1e-8)  # sqrt(lambda) * cum_signal + n
    v_tilde = trans.receiver(cum_signal, Q_level, K,
                             torch.tensor(lambda_r).cuda(),
                             v_min, v_max, flag)

    if isinstance(signal_send[0], list):
        # 将处理后的向量恢复成原始形状
        restored_signal = []
        index = 0
        for tensor in signal_send[0]:
            numel = tensor.numel()
            restored_grad = v_tilde[index:index + numel].reshape(tensor.shape)
            restored_signal.append(restored_grad)
            index += numel
    else:
        restored_signal = v_tilde
    return numpy.array(restored_signal)


def aggregate_A_Air(signal_send, v_min, v_max, K, lambda_r):
    cum_signal = torch.cat([(torch.zeros_like(grad)).flatten() for grad in signal_send[0]])

    for m_device in range(K):
        # 压缩成一维向量
        flat_grad = torch.cat([tensor.flatten() for tensor in signal_send[m_device]])
        # 归一化并求和
        cum_signal += -1 + 2 * (flat_grad - v_min) / (v_max - v_min)
    n_r = torch.normal(0, torch.sqrt(torch.tensor(1e-8)), size=cum_signal.size()).cuda()
    m_lambda = torch.sqrt(torch.tensor(lambda_r)).cuda()
    # cum_signal = m_lambda * cum_signal + n_r
    # sqrt(lambda) * cum_signal + n
    v_tilde = K * v_min + (v_max - v_min) / 2 * (cum_signal + n_r / m_lambda + K)

    if isinstance(signal_send[0], list):
        # 将处理后的向量恢复成原始形状
        restored_signal = []
        index = 0
        for tensor in signal_send[0]:
            numel = tensor.numel()
            restored_grad = v_tilde[index:index + numel].reshape(tensor.shape)
            restored_signal.append(restored_grad)
            index += numel
    else:
        restored_signal = v_tilde
    return numpy.array(restored_signal)
