import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, n_stage, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops=[]
        for n in range(n_stage):
            if n == 0:
                n_filters_in = n_filters_in
            else:
                n_filters_in = n_filters_out
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpSampleBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalizatin='none'):
        super(UpSampleBlock, self).__init__()

        ops = []
        ops.append(nn.Upsample(scale_factor=stride, mode='trilinear',align_corners=False))
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        if normalizatin == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalizatin == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalizatin == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalizatin != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()

        self.block_1 = ConvBlock(2, in_channels, 32, normalization='batchnorm')
        self.block_1_dw = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block_2 = ConvBlock(2, 32, 64, normalization='batchnorm')
        self.block_2_dw = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block_3 = ConvBlock(2, 64, 128, normalization='batchnorm')
        self.block_3_dw = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block_4 = ConvBlock(2, 128, 256, normalization='batchnorm')

    def forward(self, x):
        conv1 = self.block_1(x)
        pool1 = self.block_1_dw(conv1)

        conv2 = self.block_2(pool1)
        pool2 = self.block_2_dw(conv2)

        conv3 = self.block_3(pool2)
        pool3 = self.block_3_dw(conv3)

        conv4 = self.block_4(pool3)

        return conv4, conv3, conv2, conv1


class Adapt_dilated_conv(nn.Module):
    def __init__(self,n_stage, n_filters_in, n_filters_out, normalization='none'):
        super(Adapt_dilated_conv, self).__init__()
        self.conv_1x1_1 = nn.Conv3d(n_filters_in, n_filters_out, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm3d(n_filters_out)
        self.relu_conv_1x1_1 = nn.ReLU()

        self.conv_3x3_1 = nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bn_conv_3x3_1 = nn.BatchNorm3d(n_filters_out)
        self.relu_conv_3X3_1 = nn.ReLU()

        self.conv_3x3_2 = nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, stride=1, padding=2, dilation=2)
        self.bn_conv_3x3_2 = nn.BatchNorm3d(n_filters_out)
        self.relu_conv_3X3_2 = nn.ReLU()

        self.conv_3x3_3 = nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, stride=1, padding=5, dilation=5)
        self.bn_conv_3x3_3 = nn.BatchNorm3d(n_filters_out)
        self.relu_conv_3X3_3 = nn.ReLU()

        self.offset_conv = nn.Conv3d(n_filters_in, 4, kernel_size=3, stride=1, padding=1)
        self.offset_softmax = nn.Softmax(dim=1)
        self.offset_sigmoid = nn.Sigmoid()

        self.conv = nn.Conv3d(n_filters_out*4, n_filters_out, kernel_size=1)
        self.bn = nn.BatchNorm3d(n_filters_out)
        self.relu = nn.ReLU()


    def forward(self, x):
        x_1x1_1 = self.conv_1x1_1(x)
        x_1x1_1 = self.bn_conv_1x1_1(x_1x1_1)
        x_1x1_1 = self.relu_conv_1x1_1(x_1x1_1)

        x_3x3_1 = self.conv_3x3_1(x)
        x_3x3_1 = self.bn_conv_3x3_1(x_3x3_1)
        x_3x3_1 = self.relu_conv_1x1_1(x_3x3_1)

        x_3x3_2 = self.conv_3x3_2(x)
        x_3x3_2 = self.bn_conv_3x3_2(x_3x3_2)
        x_3x3_2 = self.relu_conv_1x1_1(x_3x3_2)

        x_3x3_3 = self.conv_3x3_3(x)
        x_3x3_3 = self.bn_conv_3x3_3(x_3x3_3)
        x_3x3_3 = self.relu_conv_1x1_1(x_3x3_3)

        offset = self.offset_conv(x)
        offset = self.offset_sigmoid(offset)

        x_1x1_1 = offset[:,0,:,:,:].unsqueeze(1)*x_1x1_1
        x_3x3_1 = offset[:,1,:,:,:].unsqueeze(1)*x_3x3_1
        x_3x3_2 = offset[:,2,:,:,:].unsqueeze(1)*x_3x3_2
        x_3x3_3 = offset[:,3,:,:,:].unsqueeze(1)*x_3x3_3

        out = torch.cat([x_1x1_1, x_3x3_1, x_3x3_2, x_3x3_3], 1)
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)

        return out, offset


class Decoder_seg(nn.Module):
    def __init__(self, out_channels):#, mode=None
        super(Decoder_seg, self).__init__()
        self.block_5_up = UpSampleBlock(256, 128, normalizatin='batchnorm')
        self.block_5 = Adapt_dilated_conv(2, 256, 128, normalization='none')

        self.block_6_up = UpSampleBlock(128, 64, normalizatin='batchnorm')
        self.block_6 = Adapt_dilated_conv(2, 128, 64, normalization='none')

        self.block_7_up = UpSampleBlock(64, 32, normalizatin='batchnorm')
        self.block_7 = Adapt_dilated_conv(2, 64, 32, normalization='none')

        # self.mode = mode
        self.block_8_seg = nn.Conv3d(32, out_channels, kernel_size=1)

    def forward(self, conv4, conv3, conv2, conv1):#
        up5 = self.block_5_up(conv4)
        up5 = torch.cat((up5, conv3), 1)
        conv5, offset_5 = self.block_5(up5)
        conv5 = nn.Dropout(0.5)(conv5)

        up6 = self.block_6_up(conv5)
        up6 = torch.cat((up6, conv2), 1)
        conv6, offset_6 = self.block_6(up6)
        conv6 = nn.Dropout(0.5)(conv6)

        up7 = self.block_7_up(conv6)
        up7 = torch.cat((up7, conv1), 1)
        conv7, offset_7 = self.block_7(up7)
        conv7 = nn.Dropout(0.5)(conv7)

        out = self.block_8_seg(conv7)

        offset_5 = F.interpolate(offset_5, size=offset_7.size()[2:], mode='trilinear', align_corners=True)
        offset_6 = F.interpolate(offset_6, size=offset_7.size()[2:], mode='trilinear', align_corners=True)
        offset = torch.cat([offset_5, offset_6, offset_7],dim=1)

        return out, offset

class EDSRConv(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EDSRConv, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv3d(in_ch, out_ch, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv3d(out_ch, out_ch, 3, padding=1),
            )

        self.residual_upsampler = torch.nn.Sequential(
            torch.nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False),
            )


    def forward(self, input):
        return self.conv(input)+self.residual_upsampler(input)

class Decoder_sr(nn.Module):
    def __init__(self):
        super(Decoder_sr, self).__init__()
        self.up_sr_2 = nn.ConvTranspose3d(32, 32, 2, stride=2)
        self.up_edsr_2 = EDSRConv(32,32)
        self.up_sr_3 = nn.ConvTranspose3d(32, 16, 2, stride=2)
        self.up_edsr_3 = EDSRConv(16,16)
        self.up_conv_last = nn.Conv3d(32,1,1)
        self.sr_decoder = Decoder_part_sr_2()

    def forward(self, conv4, conv3, conv2, conv1):
        x_sr, offset = self.sr_decoder(conv4, conv3, conv2, conv1)
        x_sr_up = self.up_sr_2(x_sr)
        x_sr_up=self.up_edsr_2(x_sr_up)
        x_sr_up = self.up_conv_last(x_sr_up)
        return x_sr_up, offset


class Decoder_part_sr_2(nn.Module):
    def __init__(self):#, mode=None
        super(Decoder_part_sr_2, self).__init__()
        self.block_5_up = UpSampleBlock(256, 128, normalizatin='batchnorm')
        self.block_5 = Adapt_dilated_conv(2, 256, 128, normalization='none')

        self.block_6_up = UpSampleBlock(128, 64, normalizatin='batchnorm')
        self.block_6 = Adapt_dilated_conv(2, 128, 64, normalization='none')

        self.block_7_up = UpSampleBlock(64, 32, normalizatin='batchnorm')
        self.block_7 = Adapt_dilated_conv(2, 64, 32, normalization='none')

    def forward(self, conv4, conv3, conv2, conv1):#
        up5 = self.block_5_up(conv4)
        up5 = torch.cat((up5, conv3), 1)
        conv5, offset_5 = self.block_5(up5)
        conv5 = nn.Dropout(0.5)(conv5)

        up6 = self.block_6_up(conv5)
        up6 = torch.cat((up6, conv2), 1)
        conv6, offset_6 = self.block_6(up6)
        conv6 = nn.Dropout(0.5)(conv6)

        up7 = self.block_7_up(conv6)
        up7 = torch.cat((up7, conv1), 1)
        conv7, offset_7 = self.block_7(up7)
        conv7 = nn.Dropout(0.5)(conv7)

        offset_5 = F.interpolate(offset_5, size=offset_7.size()[2:], mode='trilinear', align_corners=True)
        offset_6 = F.interpolate(offset_6, size=offset_7.size()[2:], mode='trilinear', align_corners=True)
        offset = torch.cat([offset_5, offset_6, offset_7],dim=1)

        return conv7, offset


class SASR_Unet(nn.Module):

    def __init__(self, in_channels, out_channels, final_sigmoid=False, sr=True):
        super(SASR_Unet, self).__init__()

        self.encoder = Encoder(in_channels)
        self.seg_decoder = Decoder_seg(out_channels)
        self.sr_decoder = Decoder_sr()
        self.pointwise = torch.nn.Sequential(
            torch.nn.Conv3d(out_channels,1,1),
            torch.nn.BatchNorm3d(1),  #添加了BN层
            torch.nn.ReLU(inplace=True)
        )

        self.sr = sr
        if final_sigmoid:
            self.final = nn.Sigmoid()
        else:
            self.final = nn.Softmax(dim=1)

    def forward(self, x):
        [conv4, conv3, conv2, conv1] = self.encoder(x)
        x_seg, offset_seg = self.seg_decoder(conv4, conv3, conv2, conv1)
        if not self.sr:
            x_seg_up = x_seg
            x_seg_out = self.final(x_seg_up)
            return x_seg_out
        x_sr_up, offset_sr = self.sr_decoder(conv4, conv3, conv2, conv1)

        x_seg_up = F.interpolate(x_seg,size=[2*i for i in x.size()[2:]], mode='trilinear', align_corners=True)

        x_seg_out = self.final(x_seg_up)
        x_sr_up = x_sr_up

        return x_seg_out, x_sr_up, self.pointwise(x_seg_up), x_sr_up, offset_seg, offset_sr
