import torch
import torch.nn as nn

def get_str_num(value, str):
    return str.count(value,0,len(str))

def gather_bn_weights(model):
    bn_weights = []
    for layer in model.modules():
        if isinstance(layer, nn.Sequential) and "Bottleneck" in str(layer):
            for m1 in layer.modules():
                if "Bottleneck" in str(m1) and len(list(m1.named_children())) > 6 and len(
                        list(m1.named_children())) != 23 and len(list(m1.named_children())) != 36 and get_str_num(
                        "Bottleneck", str(m1)) != 8:  #
                    bn_weights.append(torch.abs(m1.bn1.weight.data).cpu().numpy())
                    bn_weights.append(torch.abs(m1.bn2.weight.data).cpu().numpy())
                    # size_list.append(m1.bn1.weight.data.abs.clone())
                    # size_list.extend(m1.bn2.weight.data.abs.clone())
    # print("bn_weights is", bn_weights)
    # bn_weights = torch.zeros(sum(size_list))
    # index = 0
    # for idx, size in zip(prune_idx, size_list):
    #     bn_weights[index:(index + size)] = module_list[idx][1].weight.data.abs().clone()
    #     index += size

    return bn_weights

class BNOptimizer():
    @staticmethod
    def updateBN(sr_flag, model, s, epoch, total_epoch):
        if sr_flag:
            s = s if epoch <= total_epoch * 0.5 else s * 0.01
            for layer in model.modules():
                if isinstance(layer, nn.Sequential) and "Bottleneck" in str(layer):
                    for m1 in layer.modules():
                        if "Bottleneck" in str(m1) and len(list(m1.named_children())) > 6 and len(list(m1.named_children())) != 23 and len(list(m1.named_children())) != 36 and get_str_num("Bottleneck", str(m1)) != 8:  #
                            m1.bn1.weight.grad.data.add_( s * torch.sign(m1.bn1.weight.data) )
                            m1.bn2.weight.grad.data.add_( s * torch.sign(m1.bn2.weight.data) )



