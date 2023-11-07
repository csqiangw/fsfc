from thop import profile
import torch

device = torch.device("cuda")

def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('模型总大小为：{:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)

def get_flops_params(model,dummy_input):
    model.cuda()
    dummy_input = dummy_input.cuda()
    flops, params = profile(model, (dummy_input,))
    print('flops: ', flops, 'params: ', params)
    print('FLOPs: %.2f M, Params: %.2f M' % (flops / 1000000.0, params / 1000000.0))