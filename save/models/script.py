from collections import OrderedDict
import torch

# 加载模型
m_model = torch.load('./models_10_m7.pth')

# 移除包含特定关键字的参数
def remove_params(state_dict, keyword):
    new_state_dict = OrderedDict()
    for name, param in state_dict.items():
        if keyword not in name:
            new_state_dict[name] = param
    return new_state_dict

# 在这里，假设你要移除所有包含 'layer4.0' 关键字的参数
keyword = 'layer4.0'
new_state_dict = remove_params(m_model, keyword)

# 创建新的空 OrderedDict，并加载移除了指定参数的状态字典
new_model_state_dict = OrderedDict()
new_model_state_dict.update(new_state_dict)

# 保存新的模型
torch.save(new_model_state_dict, './models_2.pth')
