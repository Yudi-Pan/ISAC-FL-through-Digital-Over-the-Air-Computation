import copy
import logging
import os
import time
from math import ceil
import random

import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from aggregate_Aircomp import aggregate_D_Air, aggregate_A_Air
from transceiver import generate_fading
from data_set import MyDataset
from models.Nets import *
from models.Update import LocalUpdate
from models.test import test_img
from utils.options import args_parser

args = args_parser()

beta_2 = 0.8


def seed_everything(TORCH_SEED):
    random.seed(TORCH_SEED)
    os.environ['PYTHONHASHSEED'] = str(TORCH_SEED)
    np.random.seed(TORCH_SEED)
    torch.manual_seed(TORCH_SEED)
    # torch.cuda.manual_seed_all(TORCH_SEED)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def J1(Sr, Sr_1, Qr, Hr, V):
    term2 = (theta_star / args.num_users * sum(Qr) - beta_1 * Hr) * Sr
    term3 = V * (sigma2 * args.num_users / Sr +
                 2 / (args.lr * L) * (delta2 * Sr / Sr_1 - delta1 * Sr_1 / (args.lr * Sr)))
    return term2 + term3


def D_star(Sr_1):
    term = args.num_users * f_cpu * (T_max - T_cm) - nu_2 * Sr_1
    return term / (T_0 * f_cpu + nu_1 + nu_2)


def solve_funJ1S(Q, Hr, Sr_1, V):
    S_max = int(Sr_1 + D_star(Sr_1))
    S_list = [Sr_1, S_max]
    a = (theta_star / args.num_users * sum(Q[:]) - beta_1 * Hr) + 2 * V * delta2 / (args.lr * L * Sr_1)
    # (Q0 - Q_prime_0)
    b = V * (sigma2 * args.num_users - (2 * delta1 * Sr_1) / (args.lr ** 2 * L))
    if a > 0 and b > 0:
        S_mid1 = np.floor(np.sqrt(b / a))
        S_mid2 = S_mid1 + 1
        S_list.extend([x for x in [S_mid1, S_mid2] if Sr_1 < x < S_max])
    S_list = np.array(S_list)
    value_list = []
    for x in S_list:
        value_list.append(J1(x, Sr_1, Q, Hr, V))
    min_index = np.argmin(value_list)
    Sr = S_list[min_index]
    D = Sr - Sr_1
    return D


def main_fed_online_CV(dist, large_scale_fading, beta_3, beta_4):
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.7723, 0.8303, 0.9284), (0.3916, 0.3057, 0.1893)),
    ])

    root_radar_1 = './data/spect/THREE_RADAR_3000/radar_1/'
    dataset_train_1 = MyDataset(txt=root_radar_1 + 'train_1_m7.txt', transform=data_transform)

    root_radar_2 = './data/spect/THREE_RADAR_3000/radar_2/'
    dataset_train_2 = MyDataset(txt=root_radar_2 + 'train_1_m7.txt', transform=data_transform)

    root_radar_3 = './data/spect/THREE_RADAR_3000/radar_3/'
    dataset_train_3 = MyDataset(txt=root_radar_3 + 'train_1_m7.txt', transform=data_transform)

    dataset_train_4 = MyDataset(txt=root_radar_1 + 'train_2_m7.txt', transform=data_transform)

    dataset_train_5 = MyDataset(txt=root_radar_2 + 'train_2_m7.txt', transform=data_transform)

    dataset_train_6 = MyDataset(txt=root_radar_3 + 'train_2_m7.txt', transform=data_transform)

    dataset_test = MyDataset(txt='./data/spect/THREE_RADAR_3000/' + 'test_m7.txt', transform=data_transform)
    dataset_train = [dataset_train_1, dataset_train_2, dataset_train_3, dataset_train_4, dataset_train_5,
                     dataset_train_6]

    if args.num_users == 10:
        dataset_train_7 = MyDataset(txt=root_radar_1 + 'train_1_m7.txt', transform=data_transform)

        dataset_train_8 = MyDataset(txt=root_radar_2 + 'train_1_m7.txt', transform=data_transform)

        dataset_train_9 = MyDataset(txt=root_radar_3 + 'train_1_m7.txt', transform=data_transform)

        dataset_train_10 = MyDataset(txt=root_radar_1 + 'train_2_m7.txt', transform=data_transform)

        # dataset_train_11 = MyDataset(txt=root_radar_2 + 'train_2_m7.txt', transform=data_transform)
        #
        # dataset_train_12 = MyDataset(txt=root_radar_3 + 'train_2_m7.txt', transform=data_transform)
        dataset_train.extend([dataset_train_7, dataset_train_8, dataset_train_9, dataset_train_10])

    img_size = dataset_train_1[0][0].shape
    print(img_size)

    mtime = time.strftime('%Y-%m-%dT%H-%M-%S')
    log_name = 'Fed-{}.log'.format(mtime)

    # 创建一个logger
    logger = logging.getLogger('resnet')
    logger.setLevel(logging.DEBUG)

    # 创建一个handler，用于写入日志文件

    fh = logging.FileHandler('./log/{}'.format(log_name))

    fh.setLevel(logging.DEBUG)

    # 再创建一个handler，用于输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # 定义handler的输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # 给logger添加handler
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info('fed_isac_resnet_online')
    logger.info('dist:{}'.format(dist))
    logger.info('beta_1-{},beta_3-{},beta_4-{}'.format(beta_1, beta_3, beta_4))
    # logger.info(args)

    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    net_glob = ResNet.ResNet10().to(args.device)

    # model_root = './save/models/models_10_m7.pth'
    # if os.path.exists(model_root) is False:
    #     torch.save(net_glob.state_dict(), model_root)
    # net_glob.load_state_dict(torch.load(model_root))

    net_total_params = sum(p.numel() for p in net_glob.parameters())
    print('| net_total_params: {}'.format(net_total_params))
    # print(net_glob)

    net_glob.train()

    # 初始化虚拟队列 0位置不使用
    Q = np.zeros((args.num_users, args.c_rounds + 1))  # Q_k，k=1...args.num_users
    H = np.zeros(args.c_rounds + 1)

    D = np.zeros(args.c_rounds + 1)  # 防止越界
    lambda_list = np.zeros(args.c_rounds)
    D[0] = int(D_star(0))  # D[0] is D^1.
    sensing_datasize = 0
    Q_level = 3
    logger.info('number of Q_level: {}'.format(Q_level))

    power_control = 'adaptive'

    # aggregate_type = "analog"
    aggregate_type = "digital"

    # datasize_type = "non-acc equal"
    # datasize_type = "acc equal"
    datasize_type = "dynamic"

    logger.info(f'power_control:{power_control}, aggregate_type:{aggregate_type}, datasize_type: {datasize_type}')
    # logger.info('number of users: {}'.format(args.num_users))
    for tmp_round in range(args.c_rounds):
        if datasize_type == "non-acc equal":
            sensing_datasize = int(100 * S_star / (args.c_rounds * args.num_users))
        elif datasize_type == "acc equal":
            if sensing_datasize < int(S_star / args.num_users):
                sensing_datasize += int(S_star / (args.c_rounds * args.num_users)) + 1
            else:
                pass
        elif datasize_type == "dynamic":
            if sensing_datasize < int(S_star / args.num_users):
                new_sense_size = ceil(D[tmp_round] / args.num_users)
                D[tmp_round] = new_sense_size * args.num_users
                sensing_datasize += new_sense_size
            else:
                D[tmp_round] = 0
        else:
            pass
        logger.info('Sensing_datasize: {}'.format(sensing_datasize))

        # time for sensing according to sensing_datasize
        # training
        loss_locals = []
        learnable_params_locals = []
        other_params_locals = []
        batches_tr_locals = []

        V_min = []
        V_max = []
        rho = 1 / args.num_users
        for m_device in range(args.num_users):
            local = LocalUpdate(args=args, batch_size=sensing_datasize, dataset=dataset_train[m_device],
                                datasize=sensing_datasize)
            __, loss, weight_diff = local.train_batch(net=copy.deepcopy(net_glob).to(args.device))
            gradients_dict = {}
            other_params_dict = {}
            batches_tr_dict = {}

            for k, v in weight_diff.items():
                if 'weight' in k or 'bias' in k:
                    gradients_dict[k] = v
                elif 'batches_tracked' in k:
                    batches_tr_dict[k] = v
                else:
                    other_params_dict[k] = v

            # 初始化最大值和最小值
            max_value = float('-inf')
            min_value = float('inf')

            # 遍历字典的键值对，并将值乘以 scale_factor
            for key, value in gradients_dict.items():
                gradients_dict[key] = (value / args.lr) * rho

                # 更新最大值和最小值
                max_value = max(max_value, torch.max(gradients_dict[key]).item())
                min_value = min(min_value, torch.min(gradients_dict[key]).item())

            V_min.append(min_value)
            V_max.append(max_value)

            learnable_params_locals.append(copy.deepcopy(gradients_dict))
            other_params_locals.append(copy.deepcopy(other_params_dict))
            batches_tr_locals.append(copy.deepcopy(batches_tr_dict))
            loss_locals.append(copy.deepcopy(loss))

        # 提取字典的键和值到列表
        keys_learnable = list(learnable_params_locals[0].keys())
        values_learnable = [list(diff_dict.values()) for diff_dict in learnable_params_locals]

        # CSI and Lyapunov Optimization
        h = generate_fading(large_scale_fading)
        h2_min = np.square(np.min(np.abs(h)))

        # if power_control == 'high comm power':
        #     lambda_list[tmp_round] = beta_4 * Pc_max * h2_min
        # elif power_control == 'low comm power':
        #     lambda_list[tmp_round] = beta_3 * Pc_max * h2_min
        # else:
        if sensing_datasize * args.num_users < S_star:
            lambda_list[tmp_round] = beta_3 * Pc_max * h2_min
            # exploration
        else:
            lambda_list[tmp_round] = beta_4 * Pc_max * h2_min
            # exploitation



        E_sensing = (T_0 * P_sensing) * D[tmp_round]
        E_computing = theta * f_cpu ** 2 * (nu_1 * D[tmp_round] + nu_2 * sensing_datasize * args.num_users)
        E_communication = np.zeros(args.num_users)
        if datasize_type == "dynamic":
            # 更新本轮虚拟队列
            for k in range(args.num_users):
                E_communication[k] = T_cm * lambda_list[tmp_round] / np.square(np.abs(h[k]))
                y_k = (E_sensing / args.num_users
                       + E_computing / args.num_users
                       + E_communication[k] - args.E_max / args.c_rounds)
                Q[k, tmp_round + 1] = max(float(Q[k, tmp_round] + y_k), 0)
            er = beta_1 * (S_star / args.c_rounds - D[tmp_round])
            H[tmp_round + 1] = max(float(H[tmp_round] + er), 0)

            if sensing_datasize * args.num_users < S_star:
                # 确定下一轮感知
                D[tmp_round + 1] = int(
                    solve_funJ1S(Q[:, tmp_round + 1], H[tmp_round + 1], sum(D), V))
                # 假设每轮的需求或工作负载 sum(D)=sum(D[:tmp_round-1]), 因为其它都是0;
            else:
                D[tmp_round + 1] = 0
        logger.info('lambda:{:.4e}'.format(lambda_list[tmp_round]))
        logger.info('E_sensing: {:.4f}, E_computing: {:.4f}, E_comm_sum: {:.4f},'.format(E_sensing, E_computing,
                                                                                         sum(E_communication)))

        # aggregated grad
        v_min = torch.tensor(min(V_min)).cuda()
        v_max = torch.tensor(max(V_max)).cuda()
        if aggregate_type == "analog":
            agg_values_learnable = args.lr * aggregate_A_Air(values_learnable, v_min, v_max, args.num_users,
                                                             lambda_list[tmp_round])
        else:
            agg_values_learnable = args.lr * aggregate_D_Air(values_learnable, v_min, v_max, args.num_users, Q_level,
                                                             lambda_list[tmp_round])

        other_params_avg_dict = {key: torch.zeros_like(value) for key, value in other_params_locals[0].items()}

        # 计算每个键对应值的总和并直接除以 K
        for param_dict in other_params_locals:
            for key, value in param_dict.items():
                other_params_avg_dict[key] += (value * rho)  # 累加每个字典的对应值

        # 全局网络参数
        original_state_dict = net_glob.state_dict()

        # 使用 zip() 函数将两个列表组合成一个字典
        combined_dict1 = dict(zip(keys_learnable, agg_values_learnable))
        # 更新参数
        combined_dict1.update(other_params_avg_dict)
        combined_dict1.update(batches_tr_locals[0])

        for key, param in original_state_dict.items():
            # 如果键在字典2中也存在，则将两个字典对应键的值相加并存储到结果字典中
            combined_dict1[key] = combined_dict1[key] + param
        # 将更新后的 state_dict 加载回神经网络
        net_glob.load_state_dict(combined_dict1)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        logger.info('Epoch: {}'.format(tmp_round + 1))
        logger.info('Train loss: {:.4f}'.format(loss_avg))

        # testing
        net_glob.eval()

        acc_test_1, loss_train_1 = test_img(net_glob, dataset_test, args)
        logger.info("average test acc: {:.2f}%".format(acc_test_1))

    # torch.save(net_glob.state_dict(), './save/models/models_{}.pth'.format(mtime))

    np.savez('./Q_list/Q_list_{}.npz'.format(mtime), Q=Q, H=H, D=D)

    # Shutdown logging handlers to ensure the log file is properly closed
    logger.handlers.clear()
    fh.close()
    ch.close()

    return


if __name__ == "__main__":
    seed_everything(3407)
    os.environ["CUDA_VISIBLE_DEVICES"] = "3,2,0,1"
    # read args
    f_cpu = 5 * 1e8
    theta = 1e-27
    nu_1 = 2 * 1e8
    nu_2 = 1 * 1e8

    Pc_max = 5e0  # 最大功率

    sigma2 = 10
    L = 10
    delta1 = args.lr ** 2
    delta2 = 1  # G/4

    T_max = 20
    T_0 = 0.5  # Sensing Time
    T_cm = 5  # Local Model Uploading Time

    P_sensing = 1e0  # sensing power 30dbm

    theta_star = T_0 * P_sensing + theta * f_cpu ** 2 * (nu_1 + nu_2)
    V = 1e1
    beta_1 = round(theta_star / 2, 3)  # very sensitive

    args.E_max = 4 * args.c_rounds
    S_star = args.num_users * (T_max - T_cm) * (f_cpu / nu_2)

    d0 = 1
    T0 = 1e-1
    # 生成每个设备到服务器的距离
    distances = np.random.uniform(low=100, high=120, size=args.num_users)
    # dist = np.array([132.00317979, 118.19177803, 146.01486528, 111.98386792, 119.40363638, 139.42371969])
    eta_c = 2.5
    # 计算大尺度衰落
    large_scale = np.sqrt(T0 * (d0 / distances) ** eta_c)

    main_fed_online_CV(distances, large_scale, 5e-2, 2e-1)
