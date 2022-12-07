import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 stride
                 ):
        super().__init__()

        net = list()
        net += [nn.Conv2d(in_channels, mid_channels, kernel_size=3,  stride=stride, padding=1, bias=True)]
        net += [nn.ReLU()]
        net += [nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)]

        self.net = nn.Sequential(*net)

        if stride == 1 and in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=True)

    def forward(self, x):
        skip = self.skip(x)
        x = self.net(x)
        x = x + skip
        return nn.LeakyReLU(0.2, True)(x)


class UpBlock(nn.Module):
    '''A 2d-conv upsampling block with a variety of options for upsampling, and following best practices / with
    reasonable defaults. (LeakyReLU, kernel size multiple of stride)
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 post_conv=True,
                 use_dropout=False,
                 dropout_prob=0.1,
                 norm=nn.BatchNorm2d,
                 upsampling_mode='transpose'):
        '''
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param post_conv: Whether to have another convolutional layer after the upsampling layer.
        :param use_dropout: bool. Whether to use dropout or not.
        :param dropout_prob: Float. The dropout probability (if use_dropout is True)
        :param norm: Which norm to use. If None, no norm is used. Default is Batchnorm with affinity.
        :param upsampling_mode: Which upsampling mode:
                transpose: Upsampling with stride-2, kernel size 4 transpose convolutions.
                bilinear: Feature map is upsampled with bilinear upsampling, then a conv layer.
                nearest: Feature map is upsampled with nearest neighbor upsampling, then a conv layer.
                shuffle: Feature map is upsampled with pixel shuffling, then a conv layer.
        '''
        super().__init__()

        net = list()

        if upsampling_mode == 'transpose':
            net += [nn.ConvTranspose2d(in_channels,
                                       out_channels,
                                       kernel_size=4,
                                       stride=2,
                                       padding=1,
                                       bias=True if norm is None else False)]
        elif upsampling_mode == 'bilinear':
            net += [nn.UpsamplingBilinear2d(scale_factor=2)]
            net += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same', bias=True if norm is None else False)]
        elif upsampling_mode == 'nearest':
            net += [nn.UpsamplingNearest2d(scale_factor=2)]
            net += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same', bias=True if norm is None else False)]
        elif upsampling_mode == 'shuffle':
            net += [nn.PixelShuffle(upscale_factor=2)]
            net += [
                nn.Conv2d(in_channels // 4, out_channels, kernel_size=3,
                          padding='same', bias=True if norm is None else False)]
        else:
            raise ValueError("Unknown upsampling mode!")

        if norm is not None:
            net += [norm(out_channels, affine=True)]

        net += [nn.LeakyReLU(0.2, True)]

        if use_dropout:
            net += [nn.Dropout2d(dropout_prob, False)]

        if post_conv:
            net += [nn.Conv2d(out_channels,
                              out_channels,
                              kernel_size=3,
                              padding='same',
                              bias=True if norm is None else False)]

            if norm is not None:
                net += [norm(out_channels, affine=True)]

            net += [nn.LeakyReLU(0.2, True)]

            if use_dropout:
                net += [nn.Dropout2d(0.1, False)]

        self.net = nn.Sequential(*net)

    def forward(self, x, skipped=None):
        if skipped is not None:
            # input = torch.cat([skipped, x], dim=1)
            input = skipped + x
        else:
            input = x
        return self.net(input)


class DownBlock(nn.Module):
    '''A 2D-conv downsampling block following best practices / with reasonable defaults
    (LeakyReLU, kernel size multiple of stride)
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_res):
        '''
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param prep_conv: Whether to have another convolutional layer before the downsampling layer.
        :param middle_channels: If prep_conv is true, this sets the number of channels between the prep and downsampling
                                convs.
        :param use_dropout: bool. Whether to use dropout or not.
        :param dropout_prob: Float. The dropout probability (if use_dropout is True)
        :param norm: Which norm to use. If None, no norm is used. Default is Batchnorm with affinity.
        '''
        super().__init__()

        res_blocks = [
            ResBlock(
                in_channels=in_channels,
                mid_channels=out_channels // 4,
                out_channels=out_channels,
                stride=2,
            )
        ]

        for _ in range(num_res - 1):
            res_blocks.append(
                ResBlock(
                    in_channels=out_channels,
                    mid_channels=out_channels // 4,
                    out_channels=out_channels,
                    stride=1,
                )
            )

        self.net = nn.Sequential(*res_blocks)

    def forward(self, x):
        return self.net(x)


class UnetSkipConnectionBlock(nn.Module):
    '''
    Helper class for building a 2D unet_0.1.
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 res_num,
                 upsampling_mode,
                 norm=nn.BatchNorm2d,
                 submodule=None,
                 use_dropout=False,
                 dropout_prob=0.1):
        super().__init__()

        if submodule is None:
            model = [DownBlock(in_channels, out_channels, res_num, use_dropout=use_dropout, dropout_prob=dropout_prob, norm=norm),
                     UpBlock(out_channels, in_channels, use_dropout=use_dropout, dropout_prob=dropout_prob, norm=norm,
                             upsampling_mode=upsampling_mode)]
        else:
            model = [DownBlock(in_channels, out_channels, res_num, use_dropout=use_dropout, dropout_prob=dropout_prob, norm=norm),
                     submodule,
                     UpBlock(out_channels, in_channels, use_dropout=use_dropout, dropout_prob=dropout_prob, norm=norm,
                             upsampling_mode=upsampling_mode)]

        self.model = nn.Sequential(*model)

        self.res = ResBlock(in_channels, in_channels, in_channels, 1)

    def forward(self, x):
        forward_passed = self.model(x)
        # x = self.res(x)
        return nn.LeakyReLU(0.2, True)(x + forward_passed)


# https://github.com/vsitzmann/pytorch_prototyping/blob/master/pytorch_prototyping.py
class ResUnet(nn.Module):
    '''A 2d-Unet implementation with sane defaults.
    '''

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 nf0=16,
                 num_down=4,
                 res_num=2,
                 max_channels=256,
                 use_dropout=False,
                 upsampling_mode='bilinear',
                 dropout_prob=0.1,
                 norm=nn.BatchNorm2d,
                 outermost_linear=True):
        '''
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param nf0: Number of features at highest level of U-Net
        :param num_down: Number of downsampling stages.
        :param max_channels: Maximum number of channels (channels multiply by 2 with every downsampling stage)
        :param use_dropout: Whether to use dropout or no.
        :param dropout_prob: Dropout probability if use_dropout=True.
        :param upsampling_mode: Which type of upsampling should be used. See "UpBlock" for documentation.
        :param norm: Which norm to use. If None, no norm is used. Default is Batchnorm with affinity.
        :param outermost_linear: Whether the output layer should be a linear layer or a nonlinear one.
        '''
        super().__init__()

        assert (num_down > 0), "Need at least one downsampling layer in UNet."

        # Define the in block
        self.in_layer = [nn.Conv2d(in_channels, nf0, kernel_size=3, padding='same', bias=True if norm is None else False)]
        if norm is not None:
            self.in_layer += [norm(nf0, affine=True)]
        self.in_layer += [nn.LeakyReLU(0.2, True)]

        if use_dropout:
            self.in_layer += [nn.Dropout2d(dropout_prob)]
        self.in_layer = nn.Sequential(*self.in_layer)

        self.in_res = self.out_res = ResBlock(nf0, nf0, nf0, 1)

        # Define the center UNet block
        self.unet_block = UnetSkipConnectionBlock(min(2 ** (num_down-1) * nf0, max_channels),
                                                  min(2 ** (num_down-1) * nf0, max_channels),
                                                  res_num,
                                                  use_dropout=use_dropout,
                                                  dropout_prob=dropout_prob,
                                                  norm=None,  # Innermost has no norm (spatial dimension 1)
                                                  upsampling_mode=upsampling_mode)

        for i in list(range(0, num_down - 1))[::-1]:
            self.unet_block = UnetSkipConnectionBlock(min(2 ** i * nf0, max_channels),
                                                      min(2 ** (i + 1) * nf0, max_channels),
                                                      res_num,
                                                      use_dropout=use_dropout,
                                                      dropout_prob=dropout_prob,
                                                      submodule=self.unet_block,
                                                      norm=norm,
                                                      upsampling_mode=upsampling_mode)

        # Define the out layer. Each unet_0.1 block concatenates its inputs with its outputs - so the output layer
        # automatically receives the output of the in_layer and the output of the last unet_0.1 layer.
        self.out_res = ResBlock(nf0, nf0, nf0, 1)

        self.out_layer = [nn.Conv2d(nf0,
                                    3,
                                    kernel_size=3,
                                    padding='same',
                                    bias=outermost_linear or (norm is None))]
        if not outermost_linear:
            if norm is not None:
                self.out_layer += [norm(out_channels, affine=True)]
            self.out_layer += [nn.LeakyReLU(0.2, True)]

            if use_dropout:
                self.out_layer += [nn.Dropout2d(dropout_prob)]
        self.out_layer = nn.Sequential(*self.out_layer)

        self.out_layer_weight = self.out_layer[0].weight


    def forward(self, x):
        in_layer = self.in_layer(x)
        unet = self.unet_block(in_layer)
        out_layer = self.out_layer(unet)

        return x + out_layer
