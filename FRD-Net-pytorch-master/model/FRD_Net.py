import torch
import torch.nn.functional as F
from torch import nn, Tensor
from thop import profile


# from .Weights import InitWeights_He

class DropBlock(nn.Module):
    def __init__(self, block_size: int = 5, p: float = 0.1):
        super().__init__()
        self.block_size = block_size
        self.p = p

    def calculate_gamma(self, x: Tensor) -> float:
        """计算gamma
        Args:
            x (Tensor): 输入张量
        Returns:
            Tensor: gamma
        """

        invalid = (1 - self.p) / (self.block_size ** 2)
        valid = (x.shape[-1] ** 2) / ((x.shape[-1] - self.block_size + 1) ** 2)
        return invalid * valid

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.size()
        if self.training:
            gamma = self.calculate_gamma(x)
            mask_shape = (N, C, H - self.block_size + 1, W - self.block_size + 1)
            mask = torch.bernoulli(torch.full(mask_shape, gamma, device=x.device))
            mask = F.pad(mask, [self.block_size // 2] * 4, value=0)
            mask_block = 1 - F.max_pool2d(
                mask,
                kernel_size=(self.block_size, self.block_size),
                stride=(1, 1),
                padding=(self.block_size // 2, self.block_size // 2),
            )
            x = mask_block * x * (mask_block.numel() / mask_block.sum())
        return x


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, num=None, dilation=None):
        super(Conv, self).__init__(
            # 卷积后面有BN,会被BN把bias抵消所以不设置
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, bias=False, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            DropBlock(7, num),
            nn.LeakyReLU(0.1, inplace=True)
        )


class DoubleConv(nn.Module):  # 576 288
    def __init__(self, in_channels, out_channels, num=None, dilation=None):
        super(DoubleConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = Conv(in_channels, out_channels, num, dilation)
        self.conv1 = Conv(out_channels, out_channels, num, dilation)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.conv1(x1)
        out = x2 + x1
        out = self.relu(out)
        return out


class down(nn.Module):
    def __init__(self, in_c, out_c, dp=0):
        super(down, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=2,
                      padding=0, stride=2, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        x = self.down(x)
        return x


class up(nn.Module):
    def __init__(self, in_c, out_c):
        super(up, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=2,
                               padding=0, stride=2, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1, inplace=False))

    def forward(self, x):
        x = self.up(x)
        return x


class block(nn.Module):
    def __init__(self, in_c, out_c, num=None, dilation=None, is_up=False, is_down=False):
        super(block, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.is_up = is_up
        self.is_down = is_down
        self.conv = DoubleConv(in_c, out_c, num=num, dilation=dilation)
        if self.is_up == True:
            self.up = up(out_c, out_c // 2)
        if self.is_down == True:
            self.down = down(out_c, out_c * 2)

    def forward(self, x):
        x = self.conv(x)
        if self.is_up == False and self.is_down == False:
            return x
        elif self.is_up == True and self.is_down == False:
            x_up = self.up(x)
            return x, x_up
        elif self.is_up == False and self.is_down == True:
            x_down = self.down(x)
            return x, x_down
        else:
            x_up = self.up(x)
            x_down = self.down(x)
            return x, x_up, x_down


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, bias=False, kernel_size=1),
            nn.BatchNorm2d(num_classes),
            nn.Sigmoid()
        )


class Conv1_1(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(Conv1_1, self).__init__(
            nn.Conv2d(in_channels, num_classes, bias=False, kernel_size=1),
            nn.BatchNorm2d(num_classes),
            nn.LeakyReLU(0.1, inplace=False)
        )


class FRD_Net(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 1,
                 base_c: int = 32,
                 num=0.9):
        super(FRD_Net, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num = num
        self.block11 = block(in_channels, base_c, num, dilation=1, is_up=False, is_down=True)
        self.block12 = block(base_c, base_c, num, dilation=2, is_up=False, is_down=True)
        self.block13 = block(base_c * 2, base_c, num, dilation=3, is_up=False, is_down=True)
        self.block14 = block(base_c * 2, base_c, num, dilation=2, is_up=False, is_down=True)
        self.block15 = block(base_c * 2, base_c, num, dilation=1, is_up=False, is_down=True)
        self.block16 = block(base_c * 2, base_c, num=num, dilation=3, is_up=False, is_down=False)

        self.block21 = block(base_c * 2, base_c * 2, num, dilation=2, is_up=True, is_down=True)
        self.block22 = block(base_c * 4, base_c * 2, num, dilation=3, is_up=True, is_down=True)
        self.block23 = block(base_c * 6, base_c * 2, num, dilation=2, is_up=True, is_down=True)
        self.block24 = block(base_c * 6, base_c * 2, num, dilation=1, is_up=True, is_down=True)
        self.block25 = block(base_c * 6, base_c * 2, num=num, dilation=3, is_up=False, is_down=False)

        self.block31 = block(base_c * 4, base_c * 4, 0.15, dilation=3, is_up=True, is_down=False)
        self.block32 = block(base_c * 8, base_c * 4, num, dilation=2, is_up=True, is_down=False)
        self.block33 = block(base_c * 8, base_c * 4, num, dilation=1, is_up=True, is_down=False)
        self.block34 = block(base_c * 8, base_c * 4, num=num, dilation=3, is_up=False, is_down=False)

        self.up1 = up(base_c * 4, base_c * 2)
        self.up2 = up(base_c * 2, base_c)
        self.outconv = OutConv(base_c, num_classes)
        self.conv5 = Conv1_1(in_channels, base_c)
        # self.apply(InitWeights_He)
        self.conv1_1 = Conv(base_c, base_c, num=num, dilation=1)
        self.conv2_2 = Conv(base_c, base_c, num=num, dilation=2)
        self.conv3_3 = Conv(base_c, base_c, num=num, dilation=3)
        self.conv6 = Conv1_1(base_c, base_c)
        self.conv7 = Conv1_1(base_c * 8, base_c * 4)
        self.conv8 = Conv1_1(base_c * 4, base_c * 2)
        self.conv9 = Conv1_1(base_c * 3, base_c)

    def forward(self, x):
        x_1 = self.conv5(x)
        x11, xdown11 = self.block11(x)
        x12, xdown12 = self.block12(x11)
        x21, xup21, xdown21 = self.block21(xdown11)
        x13, xdown13 = self.block13(torch.cat([x12, xup21], dim=1))
        x22, xup22, xdown22 = self.block22(torch.cat([xdown12, x21], dim=1))
        x31, xup31 = self.block31(xdown21)
        x14, xdown14 = self.block14(torch.cat([x13, xup22], dim=1))
        x23, xup23, xdown23 = self.block23(torch.cat([xdown13, x22, xup31], dim=1))
        x32, xup32 = self.block32(torch.cat([xdown22, x31], dim=1))

        x15, xdown15 = self.block15(torch.cat([x14, xup23], dim=1))
        x24, xup24, xdown24 = self.block24(torch.cat([xdown14, x23, xup32], dim=1))
        x33, xup33 = self.block33(torch.cat([xdown23, x32], dim=1))

        x16 = self.block16(torch.cat([x15, xup24], dim=1))
        x25 = self.block25(torch.cat([xdown15, x24, xup33], dim=1))
        x34 = self.block34(torch.cat([xdown24, x33], dim=1))

        x3 = self.up1(x34)
        x4 = self.conv8(torch.cat([x3, x25], dim=1))
        x5 = self.up2(x4)
        x7 = self.conv9(torch.cat([x5, x16, x_1], dim=1))
        x9 = self.conv1_1(x7)
        x10 = self.conv2_2(x7)
        x11 = self.conv3_3(x7)
        x12 = self.conv6(x7)

        out = x9 + x10 + x11 + x12

        output = self.outconv(out)
        return output

# 3层

if __name__ == "__main__":
    model = FRD_Net()
    input = torch.randn(4, 3, 400, 400)  # .to(device)
    flops, params = profile(model, inputs=(input,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
