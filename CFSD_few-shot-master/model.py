# ------------------------------------------------------------------------------
# Written by Weihao Li (liweihao21@nudt.edu.cn)
# stage 1 (Base Detector Training) updates all parameters except self.meta.parameters()
# stage 2 (CCKG Meta-Training) updates self.meta_params
# stage 3 (Few-shot Fine-Tuning) updates all parameters
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import numpy as np

BN_MOMENTUM = 0.1

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


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


class PoseMSMetaResNet(nn.Module):

    def __init__(self, num_layers, block, layers, heads, filters, min_image_size, default_resolution, down_ratio):
        self.inplanes = 64
        self.deconv_with_bias = False
        self.heads = heads

        super(PoseMSMetaResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # deconv layers
        self.deconv_layers = self._make_deconv_layer(
            3,
            [256, 128, 64],
            [4, 4, 4],
        )

        # reweight
        block_meta, layers_meta = resnet_spec[num_layers]
        meta_layer = CCKG(
            block_meta, layers_meta,
            feat_dim=64,
            in_channels=3,
            out_channels=[self.heads['hm'], self.heads['wid'], self.heads['len']],
        )

        in_channels_ori = int(np.ceil(
            default_resolution[0] / min_image_size) * min_image_size / down_ratio * 64)
        wid = []
        in_channels = in_channels_ori
        for i in range(2):
            wid.append(nn.Conv1d(in_channels=in_channels, out_channels=filters[i],
                                 kernel_size=3, padding=1))
            wid.append(nn.BatchNorm1d(filters[i], momentum=BN_MOMENTUM))
            wid.append(nn.ReLU(inplace=True))
            in_channels = filters[i]
        self.wid = nn.Sequential(*wid)
        len = []
        in_channels = in_channels_ori
        for i in range(2):
            len.append(nn.Conv1d(in_channels=in_channels, out_channels=filters[i],
                                 kernel_size=3, padding=1))
            len.append(nn.BatchNorm1d(filters[i], momentum=BN_MOMENTUM))
            len.append(nn.ReLU(inplace=True))
            in_channels = filters[i]
        self.len = nn.Sequential(*len)
        reg = []
        in_channels = in_channels_ori
        for i in range(2):
            reg.append(nn.Conv1d(in_channels=in_channels, out_channels=filters[i],
                                 kernel_size=3, padding=1))
            reg.append(nn.BatchNorm1d(filters[i], momentum=BN_MOMENTUM))
            reg.append(nn.ReLU(inplace=True))
            in_channels = filters[i]
        reg.append(nn.Conv1d(in_channels=in_channels, out_channels=self.heads['reg'], kernel_size=1, padding=0))
        self.reg = nn.Sequential(*reg)
        hm = []
        in_channels = in_channels_ori
        for i in range(2):
            hm.append(nn.Conv1d(in_channels=in_channels, out_channels=filters[i],
                                kernel_size=3, padding=1))
            hm.append(nn.BatchNorm1d(filters[i], momentum=BN_MOMENTUM))
            hm.append(nn.ReLU(inplace=True))
            in_channels = filters[i]
        self.hm = nn.Sequential(*hm)

        self.meta = meta_layer

        # the parameters to updata during training stage 2:CCKG Meta-Training
        self.meta_params = list(self.meta.parameters()) + list(self.hm.parameters()) + list(self.wid.parameters()) + \
                           list(self.len.parameters()) + list(self.reg.parameters())

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x, y):
        B = x.size(0)
        C = x.size(1)  # n_cls
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        x = self.extract_features(x)
        # x = x.view(B, C, x.size(1), x.size(2), x.size(3))
        ret = {'hm': [], 'wid': [], 'len': [], 'reg': []}

        x_hm = self.hm(x)
        x_hm = x_hm.view(B, C, x_hm.size(1), 1, x_hm.size(2))
        x_wid = self.wid(x)
        x_wid = x_wid.view(B, C, x_wid.size(1), 1, x_wid.size(2))
        x_len = self.len(x)
        x_len = x_len.view(B, C, x_len.size(1), 1, x_len.size(2))
        x = x.view(B, C, x.size(1), 1, x.size(2))
        x_rw = [x_hm, x_wid, x_len]

        meta = self.meta(y, x_rw)

        for i in range(C):
            ret['hm'].append(meta[:, C * i:C * (i + 1), :self.heads['hm'], :, :])
            ret['wid'].append(meta[:, C * i:C * (i + 1), self.heads['hm']: self.heads['hm'] + self.heads['wid'], :, :])
            ret['len'].append(meta[:, C * i:C * (i + 1), self.heads['hm'] + self.heads['wid']: self.heads['hm'] + self.heads['wid'] + self.heads['len'], :, :])
            ret['reg'].append(
                self.reg(x[:, i, :, :, :].contiguous().view(x.shape[0], x.shape[2] * x.shape[3], x.shape[4])).unsqueeze(
                    -2))  # reg dose not use kernels generated by  CCKG

        ret['hm'] = torch.cat(ret['hm'], dim=2)
        ret['wid'] = torch.stack(ret['wid'], dim=2)
        ret['len'] = torch.stack(ret['len'], dim=2)
        ret['reg'] = torch.stack(ret['reg'], dim=1)

        return ret

    def extract_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        forward_conv = []
        x1 = self.layer1(x)
        forward_conv.append(x1)
        x2 = self.layer2(x1)
        forward_conv.append(x2)
        x3 = self.layer3(x2)
        forward_conv.append(x3)
        x4 = self.layer4(x3)
        x = x4

        for i in range(len(self.deconv_layers)):
            x = self.deconv_layers[i](x)
            if (i + 1) % 3 == 0:
                x = x + forward_conv[-int((i + 1) / 3)]
        x = x.contiguous().view(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        return x

    def forward_multi_class(self, x, y_codes):
        """
            x: batch of images
            y_codes: list of per-category y_code
        """

        x = self.extract_features(x)
        ret = {}
        ret['hm'] = []
        for y_code in y_codes:
            meta = self.meta.apply_code(x, y_code)
            ret['hm'].append(meta[:, :, :, :])
        ret['hm'] = torch.cat(ret['hm'], dim=1)
        ret['reg'] = self.wid(x)
        ret['reg'] = self.reg(x)

        return [ret]

    def precompute_multi_class(self, y_list):
        y_code_list = self.__getattr__('meta').extract_support_code(y_list)
        return y_code_list

    def init_weights(self, num_layers, pretrained=True):
        print('BASE => init resnet deconv weights from normal distribution')
        for _, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                # print('=> init {}.bias as 0'.format(name))
                nn.init.normal_(m.weight, std=0.001)
                if self.deconv_with_bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # print('=> init {}.weight as 1'.format(name))
                # print('=> init {}.bias as 0'.format(name))
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for _, m in self.meta.named_modules():
            if isinstance(m, nn.Conv2d):
                # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                # print('=> init {}.bias as 0'.format(name))
                nn.init.normal_(m.weight, std=0.001)
                if self.deconv_with_bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # print('=> init {}.weight as 1'.format(name))
                # print('=> init {}.bias as 0'.format(name))
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Conv1d):
                # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                # print('=> init {}.bias as 0'.format(name))
                nn.init.normal_(m.weight, std=0.001)
                if self.deconv_with_bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                # print('=> init {}.weight as 1'.format(name))
                # print('=> init {}.bias as 0'.format(name))
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for _, m in self.wid.named_modules():
            if isinstance(m, nn.Conv1d):
                # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                # print('=> init {}.bias as 0'.format(name))
                nn.init.normal_(m.weight, std=0.001)
                if self.deconv_with_bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                # print('=> init {}.weight as 1'.format(name))
                # print('=> init {}.bias as 0'.format(name))
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for _, m in self.len.named_modules():
            if isinstance(m, nn.Conv1d):
                # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                # print('=> init {}.bias as 0'.format(name))
                nn.init.normal_(m.weight, std=0.001)
                if self.deconv_with_bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                # print('=> init {}.weight as 1'.format(name))
                # print('=> init {}.bias as 0'.format(name))
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for _, m in self.reg.named_modules():
            if isinstance(m, nn.Conv1d):
                # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                # print('=> init {}.bias as 0'.format(name))
                nn.init.normal_(m.weight, std=0.001)
                if self.deconv_with_bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                # print('=> init {}.weight as 1'.format(name))
                # print('=> init {}.bias as 0'.format(name))
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        print('BASE => init final conv weights from normal distribution')
        for head in self.heads:
            if ('hm' in head):
                continue
            final_layer = self.__getattr__(head)

            for i, m in enumerate(final_layer.modules()):
                if isinstance(m, nn.Conv1d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                    # print('=> init {}.bias as 0'.format(name))
                    if m.weight.shape[0] == self.heads[head]:
                        nn.init.normal_(m.weight, std=0.001)
                        nn.init.constant_(m.bias, 0)
        # pretrained_state_dict = torch.load(pretrained)
        url = model_urls['resnet{}'.format(num_layers)]
        if pretrained:
            pretrained_state_dict = model_zoo.load_url(url)
            print('=> loading pretrained model {}'.format(url))
            self.load_state_dict(pretrained_state_dict, strict=False)


class CCKG(nn.Module):
    """
        MetaModel to predict weights of 1D convolution
        to use with precomputed feature map.
    """

    def __init__(self, block, layers,
                 feat_dim, in_channels, out_channels,
                 ):

        super(CCKG, self).__init__()
        self.inplanes = 64
        self.feat_dim = feat_dim
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=2,
                               bias=False)

        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.conv_0 = nn.Conv2d(self.inplanes, feat_dim * out_channels[0], kernel_size=1, stride=1, padding=0,
                                bias=False)
        self.conv_1 = nn.Conv2d(self.inplanes, feat_dim * out_channels[1], kernel_size=1, stride=1, padding=0,
                                bias=False)
        self.conv_2 = nn.Conv2d(self.inplanes, feat_dim * out_channels[2], kernel_size=1, stride=1, padding=0,
                                bias=False)

        self.init_weights()

        self.out_ch = out_channels

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, y, x_rw):  # y: support set, x: feature map
        o_list = []
        y = y.view(-1, y.size(2), y.size(3), y.size(4), y.size(5))  # [B*n_cls,n-shot,c,y,x]
        y_list = self.extract_support_code(y)  # for batch of support sets, [B*n_cls,c_new]
        for i in range(3):
            out_ch = self.out_ch[i]
            y_attri = y_list[i]
            x_attri = x_rw[i]
            B = x_attri.size(0)
            C = x_attri.size(1)  # n_cls
            y_attri = y_attri.view(B, C, y_attri.size(1))
            o_list.append(self.apply_code(x_attri, y_attri, out_ch))  # each corresponding image x_i to y_i
        o = torch.cat(o_list, dim=2)
        return o

    def apply_code(self, x, y_code, out_ch):  # x:[B,n_cls,C,y,x], y_code:[B,n_cls,c_new]
        batch_size = x.size(0)

        outs = []
        for xi in range(x.size(1)):
            for yi in range(y_code.size(1)):
                out = torch.nn.functional.conv1d(
                    x[:, xi, :, :, :].contiguous().view(1, batch_size * self.feat_dim * x.size(3), x.size(4)),
                    y_code[:, yi, :].contiguous().view(batch_size * out_ch, self.feat_dim * 1, 1),
                    groups=batch_size,
                )  # [1,B*256*1,x] * [B*out_ch,256*1,1] --> [B*out_ch,x]
                out = out.view(batch_size, out_ch, 1, out.size(-1))  # [B*out_ch,x] --> [B,out_ch,1,x]
                outs.append(out)
        outs = torch.stack(outs, dim=1)  # [B,n_cls,out_ch,y,x]
        return outs

    def extract_support_code(self, y):
        y_list = []
        conv_list = [self.conv_0, self.conv_1, self.conv_2]
        for i in range(3):
            yys = []
            for shot in range(y.size(1)):
                yy = self.conv1(y[:, shot, :, :, :])  # [B*n_cls,n_shot,c,y,x]-->[B*n_cls,c,y,x]
                yy = self.bn1(yy)
                yy = self.relu(yy)
                yy = self.maxpool(yy)

                yy = self.layer1(yy)
                yy = self.layer2(yy)
                yy = self.layer3(yy)
                yy = self.layer4(yy)

                yy = conv_list[i](yy)  # [B*n_cls,c_new,y,x]

                yy = torch.mean(yy.view(yy.size(0), yy.size(1), -1), dim=2)  # [B*n_cls,c_new], glob avg pool
                yys.append(yy)
            y_out = torch.mean(torch.stack(yys), dim=0)  # [n_shot,B*n_cls,c_new]-->[B*n_cls,c_new], average kernel of n-shot
            y_list.append(y_out)
        return y_list

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            print('META => loading pretrained model {}'.format(pretrained))
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            print('META => init weights from normal distribution')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.normal_(m.weight, std=0.001)
                    # nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.Conv1d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)


resnet_spec = {10: (BasicBlock, [2, 2]),
               18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def get_pose_net(num_layers, heads, head_filters, min_image_size, down_ratio, default_resolution):
    block_class, layers = resnet_spec[num_layers]

    model = PoseMSMetaResNet(num_layers, block_class, layers, heads, filters=head_filters,
                             min_image_size=min_image_size, down_ratio=down_ratio,
                             default_resolution=default_resolution)
    model.init_weights(num_layers, pretrained=False)
    return model


if __name__ == '__main__':
    import thop

    # fixed parameters
    default_resolution = [831, 1024]  # the fixed size of wideband spectrogram ([time, frequency])
    min_image_size = 32  # the shrink ratio of ResNet
    down_ratio = 4  # the shrink ratio of ResNet + deconvs

    model = get_pose_net(num_layers=18, heads={'hm': 1, 'wid': 1, 'len': 2, 'reg': 1}, head_filters=[256, 64],
                         min_image_size=min_image_size, down_ratio=down_ratio,
                         default_resolution=default_resolution)

    # a episode consisting of 2 class 3 shot
    query = torch.randn(10, 2, 3, 831, 1024)  # query wideband spectrogram (batch, n_class, channel, height, width),
    # a batch of query have n_class spectrograms, each of which contain at least one signal in n_class
    support = torch.randn(10, 2, 3, 3, 704,
                          54)  # support narrowband spectrogram (batch, n_class, n_shot, channel, height, width)
    ret = model(query, support)
