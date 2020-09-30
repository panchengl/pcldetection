import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

BN_MOMENTUM = 0.1
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, num_classes,  block, layers, **kwargs):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.conv6 = nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d( 256, 256, kernel_size=3, stride=2, padding=1)


    def forward(self, inputs):
        img_batch = inputs
        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        x0 = self.maxpool(x)

        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x2, x3, x4]

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()




def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class PrunedBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, activated_planes=None, stride=1, downsample=None):
        super(PrunedBottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, activated_planes[0])
        self.bn1 = nn.BatchNorm2d(activated_planes[0])
        self.conv2 = conv3x3(activated_planes[0], activated_planes[1], stride)
        self.bn2 = nn.BatchNorm2d(activated_planes[1])
        self.conv3 = conv1x1(activated_planes[1], planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def add_residual(self, x, y):
        y = self.conv3(y)
        y = self.bn3(y)
        if self.downsample is not None:
            x = self.downsample(x)
        y += x
        return self.relu(y)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        out = self.add_residual(identity, x)

        return out

def get_str_num(value, str):
    return str.count(value,0,len(str))

def get_activate_channel_norm(model, global_percent=0.2):
    import numpy as np
    index_weights = []
    layer_keep = 0.01
    new_indices = []
    layer_channels_weights = []
    model = model.backbone
    for layer in model.modules():
        if isinstance(layer, nn.Sequential)  and "Bottleneck" in str(layer):
            for m1 in layer.modules() :
                if "Bottleneck" in str(m1) and len(list(m1.named_children())) > 6 and len(list(m1.named_children())) != 23 and len(list(m1.named_children())) != 36 and get_str_num("Bottleneck", str(m1)) !=8: #
                    channels_1, N_1, H_1, W_1 = m1.conv1.weight.data.shape
                    channels_2, N_2, H_2, W_2 = m1.conv2.weight.data.shape
                    channels_sum_conv1 = torch.sum(m1.conv1.weight.data.abs().reshape((channels_1, -1)), dim=1)
                    channels_sum_conv2 = torch.sum(m1.conv2.weight.data.abs().reshape((channels_2, -1)), dim=1)
                    layer_channels_weights.append(channels_sum_conv1 / (N_1 * H_1 * W_1))
                    layer_channels_weights.append(channels_sum_conv2 / (N_2 * H_2 * W_2))
                    index_weights.extend(channels_sum_conv1.cpu().numpy() / (N_1 * H_1 * W_1))
                    index_weights.extend(channels_sum_conv2.cpu().numpy() / (N_2 * H_2 * W_2))
    sort_weights = np.sort(index_weights)
    # print("sorted weights is", sort_weights)
    # print("global_percent is", global_percent)
    thr_weight = sort_weights[int(len(sort_weights) * global_percent)]
    # print("thr is", thr_weight)
    for curren_layer_weights in layer_channels_weights:
        # print("curren_layer_weights is", curren_layer_weights)
        mask = curren_layer_weights.cuda().gt(thr_weight).float()
        # print("mask is", mask)
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

def get_activate_channel_sliming(model, global_percent=0.2):
    import numpy as np
    layer_keep = 0.01
    new_indices = []
    ori_channels = []
    layer_bn_weights = []
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model = model.backbone
    bn_weights = []
    for layer in model.modules():
        if isinstance(layer, nn.Sequential)  and "Bottleneck" in str(layer):
            for m1 in layer.modules() :
                if "Bottleneck" in str(m1) and len(list(m1.named_children())) > 6 and len(list(m1.named_children())) != 23 and len(list(m1.named_children())) != 36 and get_str_num("Bottleneck", str(m1)) !=8: #
                    bn_weights.extend(torch.abs(m1.bn1.weight.data).cpu().numpy())
                    layer_bn_weights.append(torch.abs(m1.bn1.weight.data) )
                    ori_channels.append(torch.abs(m1.bn1.weight.data).shape[0])
                    bn_weights.extend(torch.abs(m1.bn2.weight.data).cpu().numpy())
                    layer_bn_weights.append(torch.abs(m1.bn2.weight.data))
                    ori_channels.append(torch.abs(m1.bn2.weight.data).shape[0])
    sort_bn_weights = np.sort(bn_weights)
    # print("sorted weights is", sort_bn_weights)
    # print("global_percent is", global_percent)
    thr_bn_weight = sort_bn_weights[int(len(sort_bn_weights) * global_percent)]
    # print("thr is", thr_bn_weight)
    for curren_layer_weights in layer_bn_weights:
        # print("curren_layer_weights is", curren_layer_weights)
        mask = curren_layer_weights.cuda().gt(thr_bn_weight).float()
        # print("mask is", mask)
        min_channel_num = int(len(curren_layer_weights) * layer_keep) if int(len(curren_layer_weights) * layer_keep) > 0 else 1
        if int(torch.sum(mask)) < min_channel_num:
            _, sorted_index_weights = torch.sort(curren_layer_weights, descending=True)
            mask[sorted_index_weights[:min_channel_num]] = 1.
        new_indices.append([i for i, x in enumerate(mask.cpu().numpy()) if x == 1])
    saved_channels = []
    for indice in new_indices:
        saved_channels.append(len(indice))
    print("warming: orignal channels is : ", ori_channels)
    print("warming: saved channels is   : ", saved_channels)
    print("warming: new indices is      : ", new_indices)
    return new_indices, saved_channels



    sort_weights = np.sort(index_weights)
    # print("sorted weights is", sort_weights)
    # print("global_percent is", global_percent)
    thr_weight = sort_weights[int(len(sort_weights) * global_percent)]
    # print("thr is", thr_weight)
    for curren_layer_weights in layer_channels_weights:
        # print("curren_layer_weights is", curren_layer_weights)
        mask = curren_layer_weights.cuda().gt(thr_weight).float()
        # print("mask is", mask)
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

class PrunedResNet(nn.Module):
    def __init__(self, num_class, layers, activate_channels_list, save_channels_list):
        super(PrunedResNet, self).__init__()
        # self.activated_channels = net.activated_channels.tolist()
        self.save_channels_list = save_channels_list
        self.activate_channels_list = activate_channels_list
        self.num_class = num_class
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer_count = 0
        block = PrunedBottleneck
        self.layer1 = self._make_layer_prune(block, 64, layers[0])
        self.layer2 = self._make_layer_prune(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer_prune(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer_prune(block, 512, layers[3], stride=2)

        self.proportion = 0.5

    def _make_layer_prune(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        local_activated = self.save_channels_list[self.layer_count:self.layer_count+2]
        layers.append(block(self.inplanes, planes, local_activated, stride=stride, downsample=downsample))
        self.layer_count += 2
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            local_activated = self.save_channels_list[self.layer_count:self.layer_count+2]
            layers.append(block(self.inplanes, planes, local_activated))
            self.layer_count += 2

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x0 = self.maxpool(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x2, x3, x4]

def prune_net_parameter_init(prune_net, ori_net):
    count = 0
    import numpy as np
    indices = prune_net.backbone.activate_channels_list
    print("indices is", indices)
    if isinstance(ori_net, torch.nn.DataParallel):
        ori_net = ori_net.module
    prune_net.backbone.conv1.weight.data = ori_net.backbone.conv1.weight.data
    prune_net.backbone.bn1.weight.data = ori_net.backbone.bn1.weight.data
    prune_net.backbone.bn1.bias.data = ori_net.backbone.bn1.bias.data
    for m1, m2 in zip(prune_net.modules(), ori_net.modules()):
        if isinstance(m1, PrunedBottleneck):
            index = indices[count]
            # prune
            print("---copying block index {}...".format(count+1))
            m1.conv1.weight.data = m2.conv1.weight.data[index, :, :, :]
            m1.bn1.weight.data = m2.bn1.weight.data[index]
            m1.bn1.bias.data = m2.bn1.bias.data[index]
            count += 1

            index = indices[count]
            print("---copying block index {}...".format(count + 1))
            temp = m2.conv2.weight.data[:, indices[count-1], :, :]
            m1.conv2.weight.data = temp[index, :, :, :]
            m1.bn2.weight.data = m2.bn2.weight.data[index]
            m1.bn2.weight.data = m2.bn2.bias.data[index]
            count += 1

            m1.conv3.weight.data = m2.conv3.weight.data[:, index, :, :]
            m1.bn3.weight.data = m2.bn3.weight.data
            m1.bn3.weight.data = m2.bn3.bias.data
            if m1.downsample is not None:
                for dm1, dm2 in zip(m1.downsample.modules(), m2.downsample.modules()):
                    if isinstance(dm1, nn.Conv2d):
                        dm1.weight.data = dm2.weight.data
                    if isinstance(dm1, nn.BatchNorm2d):
                        dm1.weight.data = dm2.weight.data
                        dm1.bias.data = dm2.bias.data
            print("--------------------------")
        if isinstance(m1, nn.Linear):
            m1.weight.data = m2.weight.data
            m1.bias.data = m2.bias.data

    # for m1 in prune_net.backbone.modules():
    #     if isinstance(m1, nn.Conv2d):
    #         print(m1.weight.shape)
    return prune_net

def pruning(prune_net, ori_net):
    prune_net = prune_net_parameter_init(prune_net, ori_net)
    return prune_net

def prune_resnet_50(num_classes, activate_channels_list, save_channels_list, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # model = PrunedResNet(net, num_classes, PrunedBottleneck, [3, 4, 6, 3], **kwargs)
    model = PrunedResNet(num_classes, [3, 4, 6, 3], activate_channels_list, save_channels_list,  **kwargs)
    return model

def prune_resnet_101(num_classes, activate_channels_list, save_channels_list, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # model = PrunedResNet(net, num_classes, PrunedBottleneck, [3, 4, 6, 3], **kwargs)
    model = PrunedResNet(num_classes, [3, 4, 23, 3], activate_channels_list, save_channels_list,  **kwargs)
    return model

def prune_resnet_152(num_classes, activate_channels_list, save_channels_list, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # model = PrunedResNet(net, num_classes, PrunedBottleneck, [3, 4, 6, 3], **kwargs)
    model = PrunedResNet(num_classes, [3, 8, 36, 3], activate_channels_list, save_channels_list,  **kwargs)
    return model

def resnet18(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
    return model

if __name__ == "__main__":
    model = ResNet(5, Bottleneck, [3, 4, 6, 3])
    model.training = False
    pred = model(Variable(torch.randn(2, 3, 224, 224)))
    for i in range(len(pred)):
        print("pred_{}_shape is {}".format(i, pred[i].shape))

    # loc_preds, cls_preds = model(Variable(torch.randn(2,3,224,224)))
    # print(loc_preds.size())
    # print(cls_preds.size())
    # loc_grads = Variable(torch.randn(loc_preds.size()))
    # cls_grads = Variable(torch.randn(cls_preds.size()))
    # loc_preds.backward(loc_grads, retain_graph=True)
    # cls_preds.backward(cls_grads,retain_graph=True)