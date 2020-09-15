import torch.nn as nn
import torch

def get_activate_channel_model(model, global_percent=0.2):
    import numpy as np
    index_weights = []
    layer_keep = 0.01
    new_indices = []
    layer_channels_weights = []
    model = model.backbone
    for layer in model.modules():
        if isinstance(layer, nn.Sequential)  and "Bottleneck" in str(layer):
            for m1 in layer.modules() :
                if "Bottleneck" in str(m1) and len(list(m1.named_children())) > 6:
                    channels_1, N_1, H_1, W_1 = m1.conv1.weight.data.shape
                    channels_2, N_2, H_2, W_2 = m1.conv2.weight.data.shape
                    channels_sum_conv1 = torch.sum(m1.conv1.weight.data.abs().reshape((channels_1, -1)), dim=1)
                    channels_sum_conv2 = torch.sum(m1.conv2.weight.data.abs().reshape((channels_2, -1)), dim=1)
                    layer_channels_weights.append(channels_sum_conv1 / (N_1 * H_1 * W_1))
                    layer_channels_weights.append(channels_sum_conv2 / (N_2 * H_2 * W_2))
                    index_weights.extend(channels_sum_conv1.cpu().numpy() / (N_1 * H_1 * W_1))
                    index_weights.extend(channels_sum_conv2.cpu().numpy() / (N_2 * H_2 * W_2))
    sort_weights = np.sort(index_weights)
    thr_weight = sort_weights[int(len(sort_weights) * global_percent)]
    for curren_layer_weights in layer_channels_weights:
        mask = curren_layer_weights.cuda().gt(thr_weight).float()
        min_channel_num = int(len(curren_layer_weights) * layer_keep) if int(len(curren_layer_weights) * layer_keep) > 0 else 1
        if int(torch.sum(mask)) < min_channel_num:
            _, sorted_index_weights = torch.sort(curren_layer_weights, descending=True)
            mask[sorted_index_weights[:min_channel_num]] = 1.
        new_indices.append([i for i, x in enumerate(mask.cpu().numpy()) if x == 1])
    saved_channels = []
    for indice in new_indices:
        saved_channels.append(len(indice))
    print("warming: saved channels is: ", saved_channels)
    print("warming: new indices is: ", new_indices)
    return new_indices, saved_channels

if __name__ == "__main__":
    def climbStairs( n: int) -> int:
        if n == 0:
            return 0
        elif n == 1:
            return 1;
        elif n == 2:
            return 2;
        else:
            return climbStairs(n - 1) + climbStairs(n - 2)
    a = climbStairs(38)
    print(a)
