import random
import numpy as np
import torch
import torch.nn as nn
from models.vgg_cifar_16 import VGG

def cosine_similarity(vec1,vec2):
    return vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def k_means(data,pruned_percent,max_iter = 1000):
    print("-" * 10 + '\n' +"kmeans聚类分析")
    k = int(len(data) * pruned_percent)
    centers = {}
    n_data = data.shape[0]
    for idx,i in enumerate(random.sample(range(n_data),k)):
        centers[idx] = data[i]
    for i in range(max_iter):
        print("第{}次迭代".format(i + 1))
        clusters = {}
        clusters_idx = {}
        for j in range(k):
            clusters[j] = []
            clusters_idx[j] = []

        for sample_idx,sample in enumerate(data):
            similarity = []
            for c in centers:
                similarity.append(cosine_similarity(sample,centers[c]))
            idx = np.argmax(similarity)
            clusters[idx].append(sample)
            clusters_idx[idx].append(sample_idx)
        pre_centers = centers.copy()

        for c in clusters.keys():
            centers[c] = np.mean(clusters[c],axis=0)
        is_convergent = True
        for c in centers:
            if distance(pre_centers[c],centers[c]) != 0 :
                is_convergent = False
                break
        if is_convergent == True:
            break
    return centers,clusters,clusters_idx

def fuse(conv,bn):
    w = conv.weight
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)

    gamma = bn.weight
    beta = bn.bias

    if conv.bias is not None:
        b = conv.bias
    else:
        b = mean.new_zeros(mean.shape)

    w = w * (gamma / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
    b = (b - mean) / var_sqrt * gamma + beta

    fused_conv = nn.Conv2d(conv.in_channels,
                           conv.out_channels,
                           conv.kernel_size,
                           conv.stride,
                           conv.padding,
                           bias=True)

    fused_conv.weight = nn.Parameter(w)
    fused_conv.bias = nn.Parameter(b)
    return fused_conv

def conv_bn_fuse(bn_model):
    new_model = VGG(batch_norm=False)
    features = list(bn_model.features)
    feature_list = []
    conv = None
    bn = None
    for feature in features:
        if isinstance(feature,nn.Conv2d):
            conv = feature
        elif isinstance(feature,nn.BatchNorm2d):
            bn = feature
            fused = fuse(conv,bn)
            feature_list.append(fused)
    conv_index = 0
    linear_index = 0
    for m in new_model.modules():
        if isinstance(m,nn.Conv2d):
            m.weight = nn.Parameter(feature_list[conv_index].weight)
            m.bias = nn.Parameter(feature_list[conv_index].bias)
            conv_index += 1
        elif isinstance(m,nn.Linear):
            new_model.classifier[linear_index].weight = nn.Parameter(bn_model.classifier[linear_index].weight)
            new_model.classifier[linear_index].bias = nn.Parameter(bn_model.classifier[linear_index].bias)
            linear_index = 3
        elif isinstance(m,nn.BatchNorm1d):
            new_model.classifier[1].running_var = bn_model.classifier[1].running_var
            new_model.classifier[1].running_mean = bn_model.classifier[1].running_mean
            new_model.classifier[1].weight = nn.Parameter(bn_model.classifier[1].weight)
            new_model.classifier[1].bias = nn.Parameter(bn_model.classifier[1].bias)
    torch.save(new_model, "../cifar100_models/vgg_19_fused.pt")

def pre_prune(model,prune_ratio):
    convList = []
    cfg = []
    cfg_mask = []
    layer = 0
    for m in model.modules():
        if isinstance(m,nn.Conv2d):
            conv_data = m.weight.data.clone().cpu().numpy()
            cfg_mask_child = np.zeros(conv_data.shape[0])
            for i in range(conv_data.shape[0]):
                conv_i = conv_data[i].reshape(-1)
                convList.append(conv_i)
            convListNP = np.array(convList.copy())
            centers, clusters, clusters_idx = k_means(convListNP, prune_ratio[layer])
            layer += 1
            for c in clusters.keys():
                max = None
                maxIndex = None
                for i in range(len(clusters[c])):
                    sum = np.sum(convListNP[clusters_idx[c][i]].__abs__())
                    if max == None:
                        max = sum
                        maxIndex = i
                    else:
                        if max < sum:
                            max = sum
                            maxIndex = i
                addEdConvIndex = clusters_idx[c][maxIndex]
                cfg_mask_child[addEdConvIndex] = 1
                weight_data = None
                bias_data = None
                for i in range(len(clusters[c])):
                    if weight_data == None:
                        weight_data = m.weight.data[clusters_idx[c][i]]
                    else:
                        weight_data = weight_data + m.weight.data[clusters_idx[c][i]]
                    if bias_data == None:
                        bias_data = m.bias.data[clusters_idx[c][i]]
                    else:
                        bias_data = bias_data + m.bias.data[clusters_idx[c][i]]
                m.weight.data[addEdConvIndex] = weight_data
                m.bias.data[addEdConvIndex] = bias_data
            cfg_mask.append(cfg_mask_child)
            cfg.append(len(centers))
            convList.clear()
        elif isinstance(m,nn.MaxPool2d):
            cfg.append('M')
    torch.save(model, "model save path")
    return cfg,cfg_mask

def prune(model,cfg,cfg_mask):
    prune_model = VGG(cfg = cfg,batch_norm=False)
    prune_model.cuda()
    layer_id_in_cfg = 0
    start_mask = np.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    flag_linear = True
    for [m0, m1] in zip(model.modules(), prune_model.modules()):
        if isinstance(m0, nn.Conv2d):
            idx0 = np.squeeze(np.argwhere(start_mask))
            idx1 = np.squeeze(np.argwhere(end_mask))
            w = m0.weight.data[:, idx0, :, :].clone()
            w = w[idx1, :, :, :].clone()
            m1.weight.data = w.clone()
            m1.bias.data = m0.bias.data[idx1].clone()
            layer_id_in_cfg += 1
            start_mask = end_mask
            if layer_id_in_cfg < len(cfg_mask):
                end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(start_mask))
            if flag_linear:
                w = m0.weight.data[:,idx0].clone()
                w = w[idx0,:].clone()
                m1.weight.data = w.clone()
                m1.bias.data = m0.bias.data[idx0].clone()
                flag_linear = False
            else:
                m1.weight.data = m0.weight.data[:,idx0].clone()
                m1.bias.data = m0.bias.data.clone()
        elif isinstance(m0,nn.BatchNorm1d):
            idx0 = np.squeeze(np.argwhere(start_mask))
            m1.weight.data = m0.weight.data[idx0].clone()
            m1.bias.data = m0.bias.data[idx0].clone()
            m1.running_mean = m0.running_mean[idx0].clone()
            m1.running_var = m0.running_var[idx0].clone()
    torch.save(prune_model, "model save path")