import math

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from engine import train_epoch
import torch.nn.functional as F


def sort_by_original_gradient(sum_gradients):
    # 处理单个输入数据的情况
    if not isinstance(sum_gradients, list):
        return [0]
    
    # 处理空列表的情况
    if len(sum_gradients) == 0:
        return []
    #print(sum_gradients)
    # have_sorted = []
    # reverse_dict = {}
    # for i in range(len(sum_gradients)):
    #     reverse_dict[sum_gradients[i]] = i    
    # sorted_list = deepcopy(sum_gradients)
    # sorted_index_list = []
    # for i in range(len(sorted_list)):
    #     min_index = 0
    #     min = sorted_list[0]
    #     for j in range(len(sorted_list)):
    #         if j in have_sorted: continue
    #         if sorted_list[j]<min:
    #             min_index = j
    #             min = sorted_list[j]
    #     #sorted_index_list.append(min_index)
    #     # temp = sorted_list[min_index]
    #     # sorted_list[min_index] = sorted_list[i]
    #     # sorted_list[i] = temp
    # for index in range(len(sorted_list)):
    #     sorted_index_list.append(reverse_dict[sorted_list[index]])

    sorted_index_list = [i for i, _ in sorted(enumerate(sum_gradients), key=lambda x: x[1])]


    
    return sorted_index_list
def tensor_difference(a, b):
    
    # length_a = torch.sqrt(torch.sum(torch.pow(a,2)))
    # length_b = torch.sqrt(torch.sum(torch.pow(b,2)))
    # a_dot_b = torch.sum(a*b)
    # cos_a_b = a_dot_b/(length_a*length_b)
    # #return a_dot_b/(length_a*length_b)
    # return torch.sum(torch.pow(length_b*cos_a_b-a,2))
    return torch.sqrt(torch.sum(torch.pow(a-b,2)))
def sorted_indices_2d(matrix):
    """
    返回二维列表中元素按值从小到大排序的索引顺序
    
    参数:
    matrix (list of lists): 二维列表
    
    返回:
    list of tuples: 排序后的索引列表，每个元素为 (row, col)
    """
    # 处理空输入的情况
    if not matrix or not any(matrix):
        return []
    
    # 收集所有元素及其位置
    elements = []
    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            elements.append((value, row_idx, col_idx))
    
    # 按值排序
    sorted_elements = sorted(elements, key=lambda x: x[0])
    
    # 提取排序后的位置
    return [(row, col) for _, row, col in sorted_elements]
def compute_gradient_industry(model,train_loader,device,criterion):
    model.train()
    grad_sum = 0
    grad_cat = []
    for batch_idx, (x_chunks, chunk_lens, chunk_counts, y) in enumerate(train_loader):
        x_chunks = x_chunks.to(device)
        y = y.to(device).squeeze(1)
        pred = model(x_chunks, chunk_lens, chunk_counts)
        loss = criterion(pred, y)
        loss.backward()
        parameterss = []
        grads = []
        #parameters = model.parameters()
        for param in model.parameters():
            if param.grad is not None:
                parameterss.append(torch.sum(torch.pow(param.grad,2)))
                grads.append(param.grad.view(-1))
            grad_sum = sum(parameterss)  # 计算梯度平方和
            grad_cat = torch.cat(grads)  # 梯度展开拼接
        # grad_sums.append(grad_sum)
        # grad_cats.append(grad_cat)
    # grad_sum = sum(grad_sums)
    # grad_cat = torch.cat(grad_cats)
        return grad_sum, grad_cat.cpu()
    
def compute_gradient_EEG(model,train_loader,device,criterion):
    grad_sum = 0
    grad_cat = []
    for train_data in train_loader:
        inputs, reals = train_data
        inputs = inputs.to(device)
        reals = reals.to(device)

        parameterss = []
        grads = []
        grad_cats = []
        grad_sums = []
        outputs, _ = model(inputs)
        loss = criterion(outputs, reals)
        loss.backward()
        #parameters = model.parameters()
        for param in model.parameters():
            if param.grad is not None:
                parameterss.append(torch.sum(torch.pow(param.grad,2)))
                grads.append(param.grad.view(-1))
            grad_sum = sum(parameterss)  # 计算梯度平方和
            grad_cat = torch.cat(grads)  # 梯度展开拼接
        # grad_sums.append(grad_sum)
        # grad_cats.append(grad_cat)
    # grad_sum = sum(grad_sums)
    # grad_cat = torch.cat(grad_cats)
        return grad_sum, grad_cat.cpu()
def compute_gradient(model,train_loader,device,criterion,scaler):
    grad_sum = 0 
    grad_cat = []
    for train_data in train_loader:
        inputs, reals = train_data
        inputs = inputs.to(device)
        reals = reals.to(device)
        # print(inputs[...,:3].shape)
        # inputs = torch.Tensor(np.expand_dims(inputs[:,:,:,:3], axis=-1))  # (32,12,220,1)
        # inputs = torch.Tensor(inputs[:, :, :, :in_dim])  # (32,12,220,3)
        # num_nodes = inputs.shape[2]
        # in_dim = inputs.shape[-1]
        # reals = torch.Tensor(np.expand_dims(reals[:,:,:,:3],axis=-1))  # 同上
        # print(reals.shape)
        # reals = torch.Tensor(reals)  # (32,12,220,1)
        # print(reals.shape)
        # reals = reals.to(device)
        # inputs = inputs.transpose(1, 3)
        # inputs = nn.functional.pad(inputs, (1, 0, 0, 0))
        # inputs = inputs.to(device)  # input shape: torch.Size([32, 3, 220, 13])
        # optimizer.zero_grad()
        # outputs = model(inputs)[1]  # output shape: torch.Size([32, 12, 220, 1])
        # predict = scaler.inverse_transform(outputs)
        # loss = criterion(predict, reals)
        # loss /= loss.item()
        # #print(loss)
        # #loss = torch.tensor(1.0,requires_grad=True,device=device)
        # loss.backward()
        parameterss = []
        grads = []
        grad_cats = []
        grad_sums = []
        outputs, _ = model(inputs)
        preds = scaler.inverse_transform(outputs)
        loss = criterion(preds, reals)
        loss.backward()
        #parameters = model.parameters()
        for param in model.parameters():
            if param.grad is not None:
                parameterss.append(torch.sum(torch.pow(param.grad,2)))
                grads.append(param.grad.view(-1))
            grad_sum = sum(parameterss)  # 计算梯度平方和
            grad_cat = torch.cat(grads)  # 梯度展开拼接
        # grad_sums.append(grad_sum)
        # grad_cats.append(grad_cat)
    # grad_sum = sum(grad_sums)
    # grad_cat = torch.cat(grad_cats)
        return grad_sum, grad_cat.cpu()


def get_min_gradient(sum_gradients,new_i,new_j):
    min = sum_gradients[0][0]
    min_i = 0
    min_j = 0
    for i in range(len(sum_gradients)):
        for j in range(len(sum_gradients[i])):
            if i==new_i and j==new_j:continue
            if sum_gradients[i][j]<min:
                min_i = i
                min_j = j
                min = sum_gradients[i][j]
    return min, min_i, min_j
def reverse_sort_sample_by_gradient(min_i,min_j,cat_gradients):
    sample_order = []
    min_sample = cat_gradients[min_i][min_j]
    difference_matrix = []
    difference_matrix_copy = []
    row = len(cat_gradients)
    column = len(cat_gradients[0])

    for i in range(len(cat_gradients)):
        difference_matrix.append([])
        difference_matrix_copy.append([])
        for j in range(len(cat_gradients[i])):
            difference_matrix[i].append(tensor_difference(min_sample, cat_gradients[i][j]))
            difference_matrix_copy[i].append(tensor_difference(min_sample, cat_gradients[i][j]))

    # #index = np.argmax(cos_matrix)
    # i = min_i
    # j = min_j
    # sample_order.append([i,j])
    # difference_matrix[i][j] = 1e+200
    max_value = np.max(difference_matrix_copy)
    for _ in range(row * column):
        index = np.argmax(difference_matrix)
        i = index // column
        j = index % column
        sample_order.append([i, j])
        difference_matrix[i][j] = -1e+200
        # min_sample = cat_gradients[i][j]
        # for i in range(len(cat_gradients)):
        #     for j in range(len(cat_gradients[i])):
        #         if difference_matrix[i][j]!= 1e+200:
        #             difference_matrix[i][j] = tensor_difference(min_sample, cat_gradients[i][j])
        # index = np.argmin(difference_matrix)
        # i = index // column
        # j = index % column
        # sample_order.append([i,j])
        # difference_matrix[i][j] = 1e+200
    return sample_order, difference_matrix_copy, max_value
def sort_sample_by_gradient(min_i,min_j,cat_gradients):
    sample_order = []
    min_sample = cat_gradients[min_i][min_j]
    difference_matrix = []
    difference_matrix_copy = []
    row = len(cat_gradients)
    column = len(cat_gradients[0])

    for i in range(len(cat_gradients)):
        difference_matrix.append([])
        difference_matrix_copy.append([])
        for j in range(len(cat_gradients[i])):
            difference_matrix[i].append(tensor_difference(min_sample, cat_gradients[i][j]))
            difference_matrix_copy[i].append(tensor_difference(min_sample, cat_gradients[i][j]))

    # #index = np.argmax(cos_matrix)
    # i = min_i
    # j = min_j
    # sample_order.append([i,j])
    # difference_matrix[i][j] = 1e+200
    max_value = np.max(difference_matrix_copy)
    for _ in range(row*column):
        index = np.argmin(difference_matrix)
        i = index // column
        j = index % column
        sample_order.append([i,j])
        difference_matrix[i][j] = 1e+200
        # min_sample = cat_gradients[i][j]
        # for i in range(len(cat_gradients)):
        #     for j in range(len(cat_gradients[i])):
        #         if difference_matrix[i][j]!= 1e+200:
        #             difference_matrix[i][j] = tensor_difference(min_sample, cat_gradients[i][j])
        # index = np.argmin(difference_matrix)
        # i = index // column
        # j = index % column
        # sample_order.append([i,j])
        # difference_matrix[i][j] = 1e+200
    return sample_order,difference_matrix_copy,max_value


def _as_float(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def my_sigmoid(p0,x,d_max,circle=2):
    return p0*(1-math.exp(x-d_max))
    #return 1/(circle+math.exp(-1*x))


def dropout_schedule(kind, coeff_idx, x, d_max, p0=0.1):
    """Map gradient magnitude to dropout; all non-exp schedules decrease with x."""
    if kind == 'exp':
        return my_sigmoid(p0, x, d_max)

    coeffs = {
        'linear': [
            {'a': 0.05, 'b': 0.0},
            {'a': 0.10, 'b': 0.0},
            {'a': 0.20, 'b': 0.0},
        ],
        'inverse': [
            {'a': 0.02, 'b': 0.0, 'c': 0.40},
            {'a': 0.04, 'b': 0.0, 'c': 0.40},
            {'a': 0.08, 'b': 0.0, 'c': 0.40},
        ],
        'log': [
            {'a': 0.04, 'b': 0.0, 'c': 2.00},
            {'a': 0.08, 'b': 0.0, 'c': 2.00},
            {'a': 0.16, 'b': 0.0, 'c': 2.00},
        ],
    }
    if kind not in coeffs:
        raise ValueError(f'Unsupported dropout schedule: {kind}')
    if coeff_idx < 0 or coeff_idx >= len(coeffs[kind]):
        raise ValueError(f'Unsupported coeff_idx={coeff_idx} for dropout schedule {kind}')

    x_value = _as_float(x)
    d_max_value = max(_as_float(d_max), 1e-8)
    z = np.clip(x_value / d_max_value, 0.0, 1.0)
    s = 1.0 - z
    params = coeffs[kind][coeff_idx]

    if kind == 'linear':
        return params['a'] * s + params['b']
    if kind == 'inverse':
        return params['a'] / (z + params['c']) + params['b']
    if kind == 'log':
        return params['a'] * math.log1p(params['c'] * s) + params['b']

    raise ValueError(f'Unsupported dropout schedule: {kind}')

def my_reverse(p0,x):
    return p0*(1/(x+1))
def dynamic_sigmoid(x):
    return 1/math.exp(-1*x)

def dynamic_coupling(common_model,personal_extractor,device,num_nodes,new_sample,criterion,in_dim,scaler):
    #get_scaler(new_sample)
    coupled_model = common_model
    #compute_gradient(common_model,new_sample,device,criterion,in_dim,scaler)
    sum_gradient, cat_gradient = compute_gradient(common_model, new_sample, device, criterion, in_dim, scaler)

    lambda1 = dynamic_sigmoid(-sum_gradient)
    for (coupled_param,common_param,personal_param) in zip(coupled_model.named_parameters(),common_model.named_parameters(),personal_extractor.named_parameters()):
        # personal最后两层不要用上
        if 'end_conv_2' not in coupled_param[0] and 'end_conv_3' not in coupled_param[0]:
            coupled_param[1].data = (lambda1*common_param[1]+(1-lambda1)*personal_param[1]).data
    coupled_model.dropout = 0.7
    return coupled_model

