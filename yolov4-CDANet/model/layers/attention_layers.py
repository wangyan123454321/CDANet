import torch
import torch.nn as nn
import torch.nn.functional as F
class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,#膨胀率
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_planes = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,#每个卷积核的通道数in_planes/groups
            bias=bias,
        )#定义卷积
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )#批量归一化
        self.relu = nn.ReLU() if relu else None#激活函数

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelGate(nn.Module):
    def __init__(
        self, gate_channels, reduction_ratio=16
    ):
        super(ChannelGate, self).__init__()
        self.conv1=nn.Conv2d(64,32,kernel_size=1,padding=0, bias=False)
        self.conv2=nn.Conv2d(128,64,kernel_size=1,padding=0, bias=False)
        self.conv3=nn.Conv2d(256,128,kernel_size=1,padding=0, bias=False)
        self.conv4=nn.Conv2d(512,256,kernel_size=1,padding=0, bias=False)
        
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.conv5=nn.Conv1d(1,1,kernel_size=3,padding=1, bias=False)#padding=int(kernel_size/2)
        self.conv6=nn.Conv1d(1,1,kernel_size=5,padding=2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        N,C,H,W=x.size()
        # print("channel-x",x.shape)

        if C == 64 :
            B1=self.conv1(x)
            B2=self.conv1(x)
        elif C==128 :
            B1=self.conv2(x)
            B2=self.conv2(x)
        elif C==256 :
            B1=self.conv3(x)
            B2=self.conv3(x)
        elif C==512 :
            B1=self.conv4(x)
            B2=self.conv4(x)
        else:
            print(C)
        # print("channel-B1",B1.shape)
        # print("channel-B2",B2.shape)


        D1=(B1+B2)/2
        D2=(B1-B2)/2
        # print("channel-D1",D1.shape)
        # print("channel-D2",D2.shape)

        avg_pool1=self.avg_pool(D1)
        avg_pool2=self.avg_pool(D2)
        # print("channel-avg_pool1",avg_pool1.shape)
        # print("channel-avg_pool2",avg_pool2.shape)

        if C == 64:
            E1=self.conv5(avg_pool1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            E2=self.conv5(avg_pool2.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        elif C==128 or C==256 or C==512: #C=128,256,512,k=5
            E1=self.conv6(avg_pool1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            E2=self.conv6(avg_pool2.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # print("channel-E1",E1.shape)
        # print("channel-E2",E2.shape)

        channel_att = torch.cat(
            (E1, E2),
            dim=1,
        )
        # print("channel_att",channel_att.shape)

        scale = (
            F.sigmoid(channel_att)
        )
        # print("channel-x*scale",(x*scale).shape)
        return x*scale


######spatial attention module start


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        
        self.conv1=nn.Conv2d(64,1,kernel_size=1,padding=0, bias=False)
        self.conv2=nn.Conv2d(128,1,kernel_size=1,padding=0, bias=False)
        self.conv3=nn.Conv2d(256,1,kernel_size=1,padding=0, bias=False)
        self.conv4=nn.Conv2d(512,1,kernel_size=1,padding=0, bias=False)
        k_size = 7
        self.spatial = BasicConv(
            2,
            1,
            kernel_size=k_size,
            stride=1,
            padding=(k_size - 1) // 2,
            relu=False,
        )
    def forward(self, x):
        N,C,H,W=x.size()
        if C == 64:
            B1=self.conv1(x)
            B2=self.conv1(x)

        elif C==128:
            B1=self.conv2(x)
            B2=self.conv2(x)
        elif C==256:
            B1=self.conv3(x)
            B2=self.conv3(x)

        elif C==512:
            B1=self.conv4(x)
            B2=self.conv4(x)
        # print("B1",B1.shape)

        D1=(B1+B2)/2
        D2=(B1-B2)/2
    
        # print("D1",D1.shape)
        avg1=torch.mean(D1,1)
        avg2=torch.mean(D2,1)
        avg1=torch.unsqueeze(avg1,1)
        avg2=torch.unsqueeze(avg2,1)
        # print("avg1",avg1.shape)

        x_compress=torch.cat(
            (avg1, avg2),
            dim=1,
        )
        # print("x_com",x_compress.shape)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)
        return x * scale

class CDANet(nn.Module):
    def __init__(
        self,
        gate_channels,
        reduction_ratio=16,
    
        no_spatial=False,
    ):
        super(CDANet, self).__init__()
        self.ChannelGate = ChannelGate(
            gate_channels, reduction_ratio
        )
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out=x_out + self.SpatialGate(x)
            # x_out=self.SpatialGate(x_out)
        return x_out

