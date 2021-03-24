import torch.nn as nn
from collections import OrderedDict
from typing import Union
from pac import PacConvTranspose2d
import torch
import math
from torch.nn import functional as F
from torch.autograd import Variable

torch_ver = torch.__version__[:3]

class _UpProjection(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_UpProjection, self).__init__()

        self.conv1 = nn.Conv2d(num_input_features, num_output_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_output_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(num_output_features, num_output_features,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(num_output_features)
        self.conv2 = nn.Conv2d(num_input_features, num_output_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(num_output_features)

    def forward(self, x, size):
        x = F.upsample(x, size=size, mode='bilinear')
        x_conv1 = self.relu(self.bn1(self.conv1(x)))
        bran1 = self.bn1_2(self.conv1_2(x_conv1))
        bran2 = self.bn2(self.conv2(x))
        out = self.relu(bran1 + bran2)

        return out


class E_resnet(nn.Module):

    def __init__(self, original_model):
        super(E_resnet, self).__init__()
        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.relu = original_model.relu
        self.maxpool = original_model.maxpool

        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x_block1 = self.layer1(x)
        x_block2 = self.layer2(x_block1)
        x_block3 = self.layer3(x_block2)
        x_block4 = self.layer4(x_block3)

        return x_block1, x_block2, x_block3, x_block4


class E_densenet(nn.Module):

    def __init__(self, original_model):
        super(E_densenet, self).__init__()
        self.features = original_model.features

    def forward(self, x):
        x01 = self.features[0](x)
        x02 = self.features[1](x01)
        x03 = self.features[2](x02)
        x04 = self.features[3](x03)

        x_block1 = self.features[4](x04)
        x_block1 = self.features[5][0](x_block1)
        x_block1 = self.features[5][1](x_block1)
        x_block1 = self.features[5][2](x_block1)
        x_tran1 = self.features[5][3](x_block1)

        x_block2 = self.features[6](x_tran1)
        x_block2 = self.features[7][0](x_block2)
        x_block2 = self.features[7][1](x_block2)
        x_block2 = self.features[7][2](x_block2)
        x_tran2 = self.features[7][3](x_block2)

        x_block3 = self.features[8](x_tran2)
        x_block3 = self.features[9][0](x_block3)
        x_block3 = self.features[9][1](x_block3)
        x_block3 = self.features[9][2](x_block3)
        x_tran3 = self.features[9][3](x_block3)

        x_block4 = self.features[10](x_tran3)
        x_block4 = F.relu(self.features[11](x_block4))

        return x_block1, x_block2, x_block3, x_block4


class E_senet(nn.Module):

    def __init__(self, original_model):
        super(E_senet, self).__init__()
        self.base = nn.Sequential(*list(original_model.children())[:-3])

    def forward(self, x):
        x = self.base[0](x)
        x_block1 = self.base[1](x)
        x_block2 = self.base[2](x_block1)
        x_block3 = self.base[3](x_block2)
        x_block4 = self.base[4](x_block3)

        return x_block1, x_block2, x_block3, x_block4


class D(nn.Module):

    def __init__(self, num_features=2048):
        super(D, self).__init__()
        self.conv = nn.Conv2d(num_features, num_features //
                              2, kernel_size=1, stride=1, bias=False)
        num_features = num_features // 2
        self.bn = nn.BatchNorm2d(num_features)
        self.up1 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        self.up2 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        self.up3 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        self.up4 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features // 2)

    def forward(self, x_block1, x_block2, x_block3, x_block4):
        x_d0 = F.relu(self.bn(self.conv(x_block4)))
        x_d1 = self.up1(x_d0, [x_block3.size(2), x_block3.size(3)])
        x_d2 = self.up2(x_d1, [x_block2.size(2), x_block2.size(3)])
        x_d3 = self.up3(x_d2, [x_block1.size(2), x_block1.size(3)])

        return x_d3


class MFF(nn.Module):

    def __init__(self, block_channel, num_features=64):
        super(MFF, self).__init__()

        self.up1 = _UpProjection(
            num_input_features=block_channel[0], num_output_features=16)
        self.up2 = _UpProjection(
            num_input_features=block_channel[1], num_output_features=16)
        self.up3 = _UpProjection(
            num_input_features=block_channel[2], num_output_features=16)
        self.up4 = _UpProjection(
            num_input_features=block_channel[3], num_output_features=16)
        self.conv = nn.Conv2d(
            num_features, num_features, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x_block1, x_block2, x_block3, x_block4, size):
        x_m1 = self.up1(x_block1, size)
        x_m2 = self.up2(x_block2, size)
        x_m3 = self.up3(x_block3, size)
        x_m4 = self.up4(x_block4, size)
        x = self.bn(self.conv(torch.cat((x_m1, x_m2, x_m3, x_m4), 1)))
        x = F.relu(x)

        return x


class R(nn.Module):
    def __init__(self):
        super(R, self).__init__()
        self.conv0 = nn.Conv2d(192,64,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn0 = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(64, 64,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(
            64, 1, kernel_size=5, stride=1, padding=2, bias=True)

    def forward(self, x):
        x0 = self.conv0(x)
        x0 = self.bn0(x0)
        x0 = F.relu(x0)
        x1 = self.conv1(x0)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x2 = self.conv2(x1)

        return x2


class PacJointUpsample(nn.Module):
    def __init__(self, factor, channels=1, guide_channels=12,
                 n_t_layers=1, n_g_layers=3, n_f_layers=2,
                 n_t_filters: Union[int, tuple] = 32, n_g_filters: Union[int, tuple] = 32,
                 n_f_filters: Union[int, tuple] = 32,
                 k_ch=48, f_sz_1=5, f_sz_2=5, t_bn=False, g_bn=False, u_bn=False, f_bn=False):
        super(PacJointUpsample, self).__init__()
        self.channels = channels
        self.guide_channels = guide_channels
        self.factor = factor
        self.branch_t = None
        self.branch_g = None
        self.branch_f = None
        self.k_ch = k_ch
        self.conv_gru1 = ConvGRUCell(18,18,3,cuda_flag=True)
        self.conv_gru2 = ConvGRUCell(18,18,3,cuda_flag=True)
        self.conv_gru3 = ConvGRUCell(18,18,3,cuda_flag=True)
        self.conv_gru4 = ConvGRUCell(18,18,3,cuda_flag=True)
        self.conv_gru5 = ConvGRUCell(18,18,3,cuda_flag=True)
        self.conv_gru6 = ConvGRUCell(18,18,3,cuda_flag=True)
        self.conv_gru7 = ConvGRUCell(18,18,3,cuda_flag=True)
        self.conv_gru8 = ConvGRUCell(18,18,3,cuda_flag=True)
        self.conv_gru9 = ConvGRUCell(18,18,3,cuda_flag=True)
        self.conv_gru10 = ConvGRUCell(18,18,3,cuda_flag=True)
        self.conv_gru11 = ConvGRUCell(18,18,3,cuda_flag=True)
        self.conv_gru12 = ConvGRUCell(18,18,3,cuda_flag=True)

        assert n_g_layers >= 1, 'Guidance branch should have at least one layer'
        assert n_f_layers >= 1, 'Final prediction branch should have at least one layer'
        assert math.log2(factor) % 1 == 0, 'factor needs to be a power of 2'
        assert f_sz_1 % 2 == 1, 'filter size needs to be an odd number'
        num_ups = int(math.log2(factor))  # number of 2x up-sampling operations
        pad = int(f_sz_1 // 2)

        if type(n_t_filters) == int:
            n_t_filters = (n_t_filters,) * n_t_layers
        else:
            assert len(n_t_filters) == n_t_layers

        if type(n_g_filters) == int:
            n_g_filters = (n_g_filters,) * (n_g_layers - 1)
        else:
            assert len(n_g_filters) == n_g_layers - 1

        if type(n_f_filters) == int:
            n_f_filters = (n_f_filters,) * (n_f_layers + num_ups - 1)
        else:
            assert len(n_f_filters) == n_f_layers + num_ups - 1

        t_layers = []
        n_t_channels = (channels,) + n_t_filters
        for l in range(n_t_layers):
            t_layers.append(('conv{}'.format(l + 1), nn.Conv2d(n_t_channels[l], n_t_channels[l + 1],
                                                               kernel_size=f_sz_1, padding=pad)))
            if t_bn:
                t_layers.append(('bn{}'.format(l + 1), nn.BatchNorm2d(n_t_channels[l + 1])))
            if l < n_t_layers - 1:
                t_layers.append(('relu{}'.format(l + 1), nn.ReLU()))
        self.branch_t = nn.Sequential(OrderedDict(t_layers))

        g_layers = []
        n_g_channels = (guide_channels,) + n_g_filters + (18,)
        for l in range(n_g_layers):
            g_layers.append(('conv{}'.format(l + 1), nn.Conv2d(n_g_channels[l], n_g_channels[l + 1],
                                                               kernel_size=f_sz_1, padding=pad)))
            if g_bn:
                g_layers.append(('bn{}'.format(l + 1), nn.BatchNorm2d(n_g_channels[l + 1])))
            if l < n_g_layers - 1:
                g_layers.append(('relu{}'.format(l + 1), nn.ReLU()))
        self.branch_g = nn.Sequential(OrderedDict(g_layers))

        p, op = int((f_sz_2 - 1) // 2), (f_sz_2 % 2)
        self.up_convts = nn.ModuleList()
        self.up_bns = nn.ModuleList()
        n_f_channels = (n_t_channels[-1],) + n_f_filters + (1,)
        for l in range(num_ups):
            self.up_convts.append(PacConvTranspose2d(n_f_channels[l], n_f_channels[l + 1],
                                                     kernel_size=f_sz_2, stride=2, padding=p, output_padding=op))
            if u_bn:
                self.up_bns.append(nn.BatchNorm2d(n_f_channels[l + 1]))

        f_layers = []
        for l in range(n_f_layers):
            f_layers.append(('conv{}'.format(l + 1), nn.Conv2d(n_f_channels[l + num_ups], n_f_channels[l + num_ups + 1],
                                                               kernel_size=f_sz_1, padding=pad)))
            if f_bn:
                f_layers.append(('bn{}'.format(l + 1), nn.BatchNorm2d(n_f_channels[l + num_ups + 1])))
            if l < n_f_layers - 1:
                f_layers.append(('relu{}'.format(l + 1), nn.ReLU()))
        self.branch_f = nn.Sequential(OrderedDict(f_layers))

    def forward(self, x, guide):
        ch0 = 1
        x = self.branch_t(x)
        guide = self.branch_g(guide)
        guide_1, guide_2, guide_3, guide_4, guide_5, guide_6, guide_7, guide_8, guide_9, guide_10, guide_11, guide_12 = torch.chunk(guide, 12, dim=0)

        guide1 = self.conv_gru1(guide_1, None)
        guide2 = self.conv_gru2(guide_2, guide1)
        guide3 = self.conv_gru3(guide_3, guide2)
        guide4 = self.conv_gru4(guide_4, guide3)
        guide5 = self.conv_gru5(guide_5, guide4)
        guide6 = self.conv_gru6(guide_6, guide5)
        guide7 = self.conv_gru7(guide_7, guide6)
        guide8 = self.conv_gru8(guide_8, guide7)
        guide9 = self.conv_gru9(guide_9, guide8)
        guide10 = self.conv_gru10(guide_10, guide9)
        guide11 = self.conv_gru11(guide_11, guide10)
        guide = self.conv_gru12(guide_12, guide11)

        for i in range(len(self.up_convts)):
            scale = math.pow(2, i + 1) / self.factor
            guide_cur = guide[:, (i * self.k_ch):((i + 1) * self.k_ch)]
            if scale != 1:
                guide_cur = F.interpolate(guide_cur, scale_factor=scale, align_corners=False, mode='bilinear')
            guide_cur = repeat_for_channel(guide_cur, ch0)
            x = self.up_convts[i](x, guide_cur)
            if self.up_bns:
                x = self.up_bns[i](x)
            x = F.relu(x)

        x = self.branch_f(x)
        return x


def repeat_for_channel(x, ch):
    if ch != 1:
        bs, _ch, h, w = x.shape
        x = x.repeat(1, ch, 1, 1).reshape(bs * ch, _ch, h, w)
    return x


class PacJointUpsampleLite(PacJointUpsample):
    def __init__(self, factor, channels=1, guide_channels=3):
        if factor == 4:
            args = dict(n_g_filters=(12,22 ), n_t_filters=(22), n_f_filters=(12, 16, 22), k_ch=9)
        elif factor == 8:
            args = dict(n_g_filters=(12, 16), n_t_filters=(12, 16, 16), n_f_filters=(12, 16, 16, 20), k_ch=12)
        elif factor == 16:
            # args = dict(n_g_filters=(8, 16), n_t_filters=(8, 16, 16), n_f_filters=(8, 16, 16, 16, 16), k_ch=3)
            args = dict(n_g_filters=(8, 16), n_t_filters=(16), n_f_filters=(8, 16, 16, 16, 16), k_ch=3)
        else:
            raise ValueError('factor can only be 4, 8, or 16.')
        super(PacJointUpsampleLite, self).__init__(factor, channels, guide_channels, **args)


class ConvGRUCell(nn.Module):

    def __init__(self, input_size, hidden_size, kernel_size, cuda_flag):
        super(ConvGRUCell, self).__init__()
        self.input_size = input_size
        self.cuda_flag = cuda_flag
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.ConvGates = nn.Conv2d(self.input_size + self.hidden_size, 2 * self.hidden_size, 3,
                                   padding=self.kernel_size // 2)
        self.Conv_ct = nn.Conv2d(self.input_size + self.hidden_size, self.hidden_size, 3, padding=self.kernel_size // 2)
        dtype = torch.FloatTensor
        self.gn=nn.GroupNorm(6, self.hidden_size, eps=1e-05, affine=True)
        self.aspp1 = nn.Conv2d( self.input_size + self.hidden_size,  2 * self.hidden_size,  3, padding=3, dilation=3)
        self.aspp2 =nn.Conv2d(self.input_size + self.hidden_size,2 * self.hidden_size, 3, padding=5, dilation=5)

    def forward(self, input, hidden):
        if hidden is None:
            size_h = [input.data.size()[0], self.hidden_size] + list(input.data.size()[2:])
            if self.cuda_flag == True:
                hidden = Variable(torch.zeros(size_h)).cuda()
            else:
                hidden = Variable(torch.zeros(size_h))
        c1 = self.ConvGates(torch.cat((input, hidden), 1))
        c2 = self.aspp1(torch.cat((input, hidden), 1))
        c3 = self.aspp2(torch.cat((input, hidden), 1))
        c1 = c1 + c2 + c3
        (rt, ut) = c1.chunk(2, 1)
        rt=self.gn(rt)
        ut = self.gn(ut)
        reset_gate = F.sigmoid(rt)
        update_gate = F.sigmoid(ut)
        gated_hidden = torch.mul(reset_gate, hidden)
        p1 = self.Conv_ct(torch.cat((input, gated_hidden), 1))
        ct = F.tanh(p1)
        next_h = torch.mul(update_gate, hidden) + (1 - update_gate) * ct

        return next_h


