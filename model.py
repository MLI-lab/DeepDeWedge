"""
This is a PyTorch re-implementation of the 3D U-Net used in the IsoNet software package, which can be found here: https://github.com/IsoNet-cryoET/IsoNet/tree/master/models/unet
"""
import torch
from torch import nn

class Unet3D(torch.nn.Module):
    def __init__(
        self,
        in_chans: int = 1,
        out_chans: int = 1,
        chans: int = 32,
        num_pool_layers: int = 3,
        drop_prob: float = 0.0,
        residual: bool = True,
        normalization_loc: float = 0.0,
        normalization_scale: float = 1.0,
    ):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.residual = residual
        self.normalization_loc = nn.parameter.Parameter(torch.tensor(normalization_loc), requires_grad=False)
        self.normalization_scale = nn.parameter.Parameter(torch.tensor(normalization_scale), requires_grad=False)

        self.down_sample_layers = nn.ModuleList([DownConvBlock(in_chans, chans, drop_prob)])
        self.down_samplers = nn.ModuleList([SpatialDownSampling(chans)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(DownConvBlock(ch, ch * 2, drop_prob))
            self.down_samplers.append(SpatialDownSampling(ch * 2))
            ch *= 2
        
        self.bottleneck = nn.Sequential(
            nn.Conv3d(ch, ch * 2, kernel_size=(3, 3, 3), padding=1),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Conv3d(ch * 2, ch, kernel_size=(3, 3, 3), padding=1),
        )

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList([SpatialUpSampling(in_chans=ch, out_chans=ch)])
        for _ in range(num_pool_layers-1):
            self.up_conv.append(UpConvBlock(2 * ch, ch, drop_prob))
            self.up_transpose_conv.append(SpatialUpSampling(in_chans=ch, out_chans=ch//2))
            ch //= 2
        self.up_conv.append(UpConvBlock(2 * ch, ch, drop_prob))
        self.final_conv = nn.Conv3d(ch, self.out_chans, kernel_size=(1, 1, 1), stride=(1, 1, 1))

    def normalize(self, volume: torch.Tensor) -> torch.Tensor:
        return (volume - self.normalization_loc) / self.normalization_scale

    def denormalize(self, volume: torch.Tensor) -> torch.Tensor:
        return volume * self.normalization_scale + self.normalization_loc

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        volume = self.normalize(volume)

        stack = []
        output = volume

        # apply down-sampling layers
        for layer, downsampler in zip(self.down_sample_layers, self.down_samplers):
            output = layer(output)
            stack.append(output)
            output = downsampler(output)
            
        output = self.bottleneck(output)

        # apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output, cat=downsample_layer)
            output = conv(output)

        output = self.final_conv(output)
        if self.residual:
            output = output + volume

        output = self.denormalize(output)
        return output


class DownConvBlock(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv3d(in_chans, out_chans, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(out_chans),
            nn.Dropout3d(drop_prob),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Conv3d(out_chans, out_chans, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(out_chans),
            nn.Dropout3d(drop_prob),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Conv3d(out_chans, out_chans, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(out_chans),
            nn.Dropout3d(drop_prob),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
        )

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        return self.layers(volume)


class UpConvBlock(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv3d(in_chans, in_chans//2, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(in_chans//2),
            nn.Dropout3d(drop_prob),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Conv3d(in_chans//2, in_chans//2, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(in_chans//2),
            nn.Dropout3d(drop_prob),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Conv3d(in_chans//2, out_chans, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(out_chans),
            nn.Dropout3d(drop_prob),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
        )

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        return self.layers(volume)

class SpatialDownSampling(nn.Module):
    def __init__(self, chans: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(chans, chans, kernel_size=(3,3,3), stride=(2,2,2), padding=1),
            nn.LeakyReLU(negative_slope=0.05, inplace=True)
        )
    
    def forward(self, volume):
        return self.layers(volume)


class SpatialUpSampling(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, drop_prob=0.):
        super().__init__()
        self.tconv = nn.ConvTranspose3d(
            in_chans, out_chans, kernel_size=(3,3,3), stride=(2,2,2), padding=1, output_padding=1
        )
        self.activation = nn.LeakyReLU(negative_slope=0.05, inplace=True)


    def forward(self, volume: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        output = self.tconv(volume)
        output = torch.cat([output, cat], dim=1)
        output = self.activation(output)
        return output