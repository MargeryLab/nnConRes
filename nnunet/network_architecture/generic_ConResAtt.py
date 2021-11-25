#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from copy import deepcopy
from nnunet.utilities.nd_softmax import sigmoid_helper
from torch import nn
import torch
import numpy as np
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
import torch.nn.functional
from torch.nn import functional as F


class ConvDropoutNormNonlin(nn.Module):
    """
    fixes a bug in ConvDropoutNormNonlin where lrelu was used regardless of nonlin. Bad.
    """

    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(ConvDropoutNormNonlin, self).__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs[
            'p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))


class ConvDropoutNonlinNorm(ConvDropoutNormNonlin):
    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.instnorm(self.lrelu(x))


class StackedConvLayers(nn.Module):
    def __init__(self, input_feature_channels, output_feature_channels, num_convs,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, first_stride=None, basic_block=ConvDropoutNormNonlin):
        '''
        stacks ConvDropoutNormLReLU layers. initial_stride will only be applied to first layer in the stack. The other parameters affect all layers
        :param input_feature_channels:
        :param output_feature_channels:
        :param num_convs:
        :param dilation:
        :param kernel_size:
        :param padding:
        :param dropout:
        :param initial_stride:
        :param conv_op:
        :param norm_op:
        :param dropout_op:
        :param inplace:
        :param neg_slope:
        :param norm_affine:
        :param conv_bias:
        '''
        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels

        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        if first_stride is not None:
            self.conv_kwargs_first_conv = deepcopy(conv_kwargs)
            self.conv_kwargs_first_conv['stride'] = first_stride
        else:
            self.conv_kwargs_first_conv = conv_kwargs

        super(StackedConvLayers, self).__init__()
        self.blocks = nn.Sequential(
            *([basic_block(input_feature_channels, output_feature_channels, self.conv_op,
                           self.conv_kwargs_first_conv,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs)] +
              [basic_block(output_feature_channels, output_feature_channels, self.conv_op,
                           self.conv_kwargs,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs) for _ in range(num_convs - 1)]))

    def forward(self, x):
        return self.blocks(x)


def print_module_training_status(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d) or isinstance(module, nn.Dropout3d) or \
            isinstance(module, nn.Dropout2d) or isinstance(module, nn.Dropout) or isinstance(module, nn.InstanceNorm3d) \
            or isinstance(module, nn.InstanceNorm2d) or isinstance(module, nn.InstanceNorm1d) \
            or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d) or isinstance(module,
                                                                                                      nn.BatchNorm1d):
        print(str(module), module.training)


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                         align_corners=self.align_corners)

class Conv3d(nn.Conv3d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1), groups=1, bias=False):
        super(Conv3d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True)
        weight = weight - weight_mean
        std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12).view(-1, 1, 1, 1, 1)
        weight = weight / std.expand_as(weight)
        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv3x3x3(in_planes, out_planes, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1,1,1), dilation=(1,1,1), bias=False,
              weight_std=False):
    "3x3x3 convolution with padding"
    if weight_std:
        return Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      bias=bias)
    else:
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                         dilation=dilation, bias=bias)


class ConResAtt(nn.Module):
    def __init__(self, in_planes, out_planes, norm_op, norm_op_kwargs, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1),
                 dilation=(1, 1, 1), bias=False, weight_std=False, first_layer=False):
        super(ConResAtt, self).__init__()
        self.weight_std = weight_std
        self.stride = stride
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.first_layer = first_layer

        self.relu = nn.LeakyReLU(inplace=True)

        # self.gn_seg = nn.GroupNorm(8, in_planes)
        self.gn_seg = norm_op(in_planes, **norm_op_kwargs)
        self.conv_seg = conv3x3x3(in_planes, out_planes, kernel_size=(kernel_size[0], kernel_size[1], kernel_size[2]),
                               stride=(stride[0], stride[1], stride[2]), padding=(padding[0], padding[1], padding[2]),
                               dilation=(dilation[0], dilation[1], dilation[2]), bias=bias, weight_std=self.weight_std)

        # self.gn_res = nn.GroupNorm(8, out_planes)
        self.gn_res = norm_op(out_planes, **norm_op_kwargs)
        self.conv_res = conv3x3x3(out_planes, out_planes, kernel_size=(1,1,1),
                               stride=(1, 1, 1), padding=(0,0,0),
                               dilation=(dilation[0], dilation[1], dilation[2]), bias=bias, weight_std=self.weight_std)

        # self.gn_res1 = nn.GroupNorm(8, out_planes)
        self.gn_res1 = norm_op(out_planes, **norm_op_kwargs)
        self.conv_res1 = conv3x3x3(out_planes, out_planes, kernel_size=(kernel_size[0], kernel_size[1], kernel_size[2]),
                                stride=(1, 1, 1), padding=(padding[0], padding[1], padding[2]),
                                dilation=(dilation[0], dilation[1], dilation[2]), bias=bias, weight_std=self.weight_std)
        # self.gn_res2 = nn.GroupNorm(8, out_planes)
        self.gn_res2 = norm_op(out_planes, **norm_op_kwargs)
        self.conv_res2 = conv3x3x3(out_planes, out_planes, kernel_size=(kernel_size[0], kernel_size[1], kernel_size[2]),
                                stride=(1, 1, 1), padding=(padding[0], padding[1], padding[2]),
                                dilation=(dilation[0], dilation[1], dilation[2]), bias=bias, weight_std=self.weight_std)

        # self.gn_mp = nn.GroupNorm(8, in_planes)
        self.gn_mp = norm_op(in_planes, **norm_op_kwargs)
        self.conv_mp_first = conv3x3x3(1, out_planes, kernel_size=(kernel_size[0], kernel_size[1], kernel_size[2]),
                              stride=(stride[0], stride[1], stride[2]), padding=(padding[0], padding[1], padding[2]),
                              dilation=(dilation[0], dilation[1], dilation[2]), bias=bias, weight_std=self.weight_std)
        self.conv_mp = conv3x3x3(in_planes, out_planes, kernel_size=(kernel_size[0], kernel_size[1], kernel_size[2]),
                               stride=(stride[0], stride[1], stride[2]), padding=(padding[0], padding[1], padding[2]),
                               dilation=(dilation[0], dilation[1], dilation[2]), bias=bias, weight_std=self.weight_std)

        self._deep_supervision = True

    def _res(self, x):  # bs, channel, D, W, H

        bs, channel, depth, heigt, width = x.shape
        x_copy = torch.zeros_like(x).cuda()
        x_copy[:, :, 1:, :, :] = x[:, :, 0: depth - 1, :, :]
        res = x - x_copy
        res[:, :, 0, :, :] = 0
        res = torch.abs(res)
        return res

    def forward(self, input):
        seg_outputs = []
        x1, x2 = input
        if self.first_layer:
            x1 = self.gn_seg(x1)
            x1 = self.relu(x1)
            x1 = self.conv_seg(x1)#torch.Size([1, 64, 8, 40, 40])

            res = torch.sigmoid(x1)
            res = self._res(res)
            res = self.conv_res(res)

            x2 = self.conv_mp_first(x2)
            x2 = x2 + res

        else:
            x1 = self.gn_seg(x1)
            x1 = self.relu(x1)
            x1 = self.conv_seg(x1)

            res = torch.sigmoid(x1)
            res = self._res(res)
            res = self.conv_res(res)


            if self.in_planes != self.out_planes:
                x2 = self.gn_mp(x2)
                x2 = self.relu(x2)
                x2 = self.conv_mp(x2)

            x2 = x2 + res

        x2 = self.gn_res1(x2)
        x2 = self.relu(x2)
        x2 = self.conv_res1(x2)

        x1 = x1*(1 + torch.sigmoid(x2))

        return [x1, x2]


    @staticmethod
    def compute_approx_vram_consumption(patch_size, num_pool_per_axis, base_num_features, max_num_features,
                                        num_modalities, num_classes, pool_op_kernel_sizes, deep_supervision=False,
                                        conv_per_stage=2):
        """
        This only applies for num_conv_per_stage and convolutional_upsampling=True
        not real vram consumption. just a constant term to which the vram consumption will be approx proportional
        (+ offset for parameter storage)
        :param deep_supervision:
        :param patch_size:
        :param num_pool_per_axis:
        :param base_num_features:
        :param max_num_features:
        :param num_modalities:
        :param num_classes:
        :param pool_op_kernel_sizes:
        :return:
        """
        if not isinstance(num_pool_per_axis, np.ndarray):
            num_pool_per_axis = np.array(num_pool_per_axis)

        npool = len(pool_op_kernel_sizes)

        map_size = np.array(patch_size)
        tmp = np.int64((conv_per_stage * 2 + 1) * np.prod(map_size, dtype=np.int64) * base_num_features +
                       num_modalities * np.prod(map_size, dtype=np.int64) +
                       num_classes * np.prod(map_size, dtype=np.int64))

        num_feat = base_num_features

        for p in range(npool):
            for pi in range(len(num_pool_per_axis)):
                map_size[pi] /= pool_op_kernel_sizes[p][pi]
            num_feat = min(num_feat * 2, max_num_features)
            num_blocks = (conv_per_stage * 2 + 1) if p < (npool - 1) else conv_per_stage  # conv_per_stage + conv_per_stage for the convs of encode/decode and 1 for transposed conv
            tmp += num_blocks * np.prod(map_size, dtype=np.int64) * num_feat
            if deep_supervision and p < (npool - 2):
                tmp += np.prod(map_size, dtype=np.int64) * num_classes
            # print(p, map_size, num_feat, tmp)
        return tmp

class NoBottleneck(nn.Module):
    def __init__(self, inplanes, planes, norm_op, norm_op_kwargs, stride=(1, 1, 1), dilation=(1, 1, 1), downsample=None, fist_dilation=1,
                 multi_grid=1, weight_std=False):
        super(NoBottleneck, self).__init__()
        self.weight_std = weight_std
        self.relu = nn.LeakyReLU(inplace=True)

        # self.gn1 = nn.GroupNorm(8, inplanes)
        self.gn1 = norm_op(inplanes, **norm_op_kwargs)
        self.conv1 = conv3x3x3(inplanes, planes, kernel_size=(3, 3, 3), stride=stride, padding=dilation * multi_grid,
                                 dilation=dilation * multi_grid, bias=False, weight_std=self.weight_std)

        # self.gn2 = nn.GroupNorm(8, planes)
        self.gn2 = norm_op(planes, **norm_op_kwargs)
        self.conv2 = conv3x3x3(planes, planes, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=dilation * multi_grid,
                                 dilation=dilation * multi_grid, bias=False, weight_std=self.weight_std)

        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        skip = x

        seg = self.gn1(x)
        seg = self.relu(seg)
        seg = self.conv1(seg)

        seg = self.gn2(seg)
        seg = self.relu(seg)
        seg = self.conv2(seg)

        if self.downsample is not None:
            skip = self.downsample(x)

        seg = seg + skip
        return seg


class conresnet(SegmentationNetwork):
    def __init__(self, shape, conv_op=nn.Conv2d, norm_op=nn.BatchNorm2d, norm_op_kwargs=None, block=NoBottleneck, layers=[1, 2, 2, 2, 2], num_classes=3, deep_supervision=True,weight_std=True):
        self.shape = tuple(shape)
        self.weight_std = weight_std
        super(conresnet, self).__init__()

        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.num_classes = num_classes
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine':True, 'momentum':0.1}
        self.norm_op_kwargs = norm_op_kwargs

        self.conv_4_32 = nn.Sequential(
            conv3x3x3(1, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), weight_std=self.weight_std))

        self.conv_32_64 = nn.Sequential(
            # nn.GroupNorm(8, 32),
            self.norm_op(32, **self.norm_op_kwargs),
            nn.LeakyReLU(inplace=True),
            conv3x3x3(32, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), weight_std=self.weight_std))

        self.conv_64_128 = nn.Sequential(
            # nn.GroupNorm(8, 64),
            self.norm_op(64, **self.norm_op_kwargs),
            nn.LeakyReLU(inplace=True),
            conv3x3x3(64, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), weight_std=self.weight_std))

        self.conv_128_256 = nn.Sequential(
            # nn.GroupNorm(8, 128),
            self.norm_op(128, **self.norm_op_kwargs),
            nn.LeakyReLU(inplace=True),
            conv3x3x3(128, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2), weight_std=self.weight_std))

        self.layer0 = self._make_layer(block, 32, 32, layers[0], stride=(1, 1, 1))
        self.layer1 = self._make_layer(block, 64, 64, layers[1], stride=(1, 1, 1))
        self.layer2 = self._make_layer(block, 128, 128,layers[2], stride=(1, 1, 1))
        self.layer3 = self._make_layer(block, 256, 256, layers[3], stride=(1, 1, 1))
        self.layer4 = self._make_layer(block, 256, 256, layers[4], stride=(1, 1, 1), dilation=(2,2,2))

        self.fusionConv = nn.Sequential(
            # nn.GroupNorm(8, 256),
            self.norm_op(256, **self.norm_op_kwargs),
            nn.LeakyReLU(inplace=True),
            nn.Dropout3d(0.1),
            conv3x3x3(256, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), weight_std=self.weight_std)
        )

        self.seg_x4 = nn.Sequential(
            ConResAtt(128, 64, self.norm_op, self.norm_op_kwargs,kernel_size=(3, 3, 3), padding=(1, 1, 1), weight_std=self.weight_std, first_layer=True))
        self.seg_x2 = nn.Sequential(
            ConResAtt(64, 32, self.norm_op, self.norm_op_kwargs,kernel_size=(3, 3, 3), padding=(1, 1, 1), weight_std=self.weight_std))
        self.seg_x1 = nn.Sequential(
            ConResAtt(32, 32, self.norm_op, self.norm_op_kwargs, kernel_size=(3, 3, 3), padding=(1, 1, 1), weight_std=self.weight_std))

        self.seg_cls = nn.Sequential(
            nn.Conv3d(32, self.num_classes, kernel_size=1)
        )
        self.res_cls = nn.Sequential(
            nn.Conv3d(32, 2, kernel_size=1)
        )
        self.resx2_cls = nn.Sequential(
            nn.Conv3d(32, 2, kernel_size=1)
        )
        self.resx4_cls = nn.Sequential(
            nn.Conv3d(64, 2, kernel_size=1)
        )

    def _make_layer(self, block, inplanes, outplanes, blocks, stride=(1, 1, 1), dilation=(1, 1, 1), multi_grid=1):
        downsample = None
        if stride[0] != 1 or stride[1] != 1 or stride[2] != 1 or inplanes != outplanes:
            downsample = nn.Sequential(
                # nn.GroupNorm(8, inplanes),
                self.norm_op(inplanes, **self.norm_op_kwargs),
                nn.LeakyReLU(inplace=True),
                conv3x3x3(inplanes, outplanes, kernel_size=(1, 1, 1), stride=stride, padding=(0, 0, 0),
                            weight_std=self.weight_std)
            )

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(inplanes, outplanes, self.norm_op, self.norm_op_kwargs,stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid), weight_std=self.weight_std))
        for i in range(1, blocks):
            layers.append(
                block(inplanes, outplanes, self.norm_op, self.norm_op_kwargs, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid),
                      weight_std=self.weight_std))
        return nn.Sequential(*layers)


    def forward(self, x):
        cha, _, dep, hei, wei = x.shape
        shape = (dep, hei, wei)
        x_copy = torch.zeros((cha, _, dep, hei, wei), dtype=torch.float32, device='cuda')
        x_copy[:, :, 1:, :, :] = x[:, :, 0:dep-1, :, :]
        x_res = x - x_copy
        x_res[:,:, 0,:,:] = 0
        x_res = torch.abs(x_res)

        ## encoder
        x = self.conv_4_32(x)
        x = self.layer0(x)
        skip1 = x   #torch.Size([1, 32, 32, 160, 160])

        x = self.conv_32_64(x)
        x = self.layer1(x)
        skip2 = x   #torch.Size([1, 64, 16, 80, 80])

        x = self.conv_64_128(x)
        x = self.layer2(x)
        skip3 = x   #torch.Size([1, 128, 8, 40, 40])

        x = self.conv_128_256(x)
        x = self.layer3(x)  #torch.Size([1, 256, 4, 20, 20])

        x = self.layer4(x)

        x = self.fusionConv(x)  #torch.Size([1, 128, 4, 20, 20])

        ## decoder
        res_x4 = F.interpolate(x_res, size=(int(shape[0] / 4), int(shape[1] / 4), int(shape[2] / 4)), mode='trilinear', align_corners=True)#torch.Size([1, 1, 8, 40, 40])
        seg_x4 = F.interpolate(x, size=(int(shape[0] / 4), int(shape[1] / 4), int(shape[2] / 4)), mode='trilinear', align_corners=True)#torch.Size([1, 128, 8, 40, 40])
        seg_x4 = seg_x4 + skip3 #torch.Size([1, 128, 8, 40, 40])
        seg_x4, res_x4 = self.seg_x4([seg_x4, res_x4])

        res_x2 = F.interpolate(res_x4, size=(int(shape[0] / 2), int(shape[1] / 2), int(shape[2] / 2)), mode='trilinear', align_corners=True)#torch.Size([1, 64, 8, 80, 80])
        seg_x2 = F.interpolate(seg_x4, size=(int(shape[0] / 2), int(shape[1] / 2), int(shape[2] / 2)), mode='trilinear', align_corners=True)
        seg_x2 = seg_x2 + skip2
        seg_x2, res_x2 = self.seg_x2([seg_x2, res_x2])

        res_x1 = F.interpolate(res_x2, size=(int(shape[0] / 1), int(shape[1] / 1), int(shape[2] / 1)), mode='trilinear', align_corners=True)
        seg_x1 = F.interpolate(seg_x2, size=(int(shape[0] / 1), int(shape[1] / 1), int(shape[2] / 1)), mode='trilinear', align_corners=True)
        seg_x1 = seg_x1 + skip1
        seg_x1, res_x1 = self.seg_x1([seg_x1, res_x1])

        seg = self.seg_cls(seg_x1)
        res = self.res_cls(res_x1)
        resx2 = self.resx2_cls(res_x2)
        resx4 = self.resx4_cls(res_x4)

        resx2 = F.interpolate(resx2, size=(int(shape[0] / 1), int(shape[1] / 1), int(shape[2] / 1)),
                      mode='trilinear', align_corners=True)
        resx4 = F.interpolate(resx4, size=(int(shape[0] / 1), int(shape[1] / 1), int(shape[2] / 1)),
                      mode='trilinear', align_corners=True)
        if self.do_ds:
            return [seg, res, resx2, resx4]
        else:
            return seg