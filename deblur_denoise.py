import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
import os
import skimage.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
from models import Unet, ResUnet
import time
from torch import nn
from torchvision.models import vgg16
from torch.fft import fft2, ifft2
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Lambda
from denoise import BSDS300Dataset, add_noise, img_to_numpy, calc_psnr, PerceptualLoss

# set random seeds
torch.manual_seed(1)
torch.use_deterministic_algorithms(True)

matplotlib.rcParams['figure.raise_window'] = False

device = torch.device("mps")


def psf2otf(psf, shape):
    inshape = psf.shape
    psf = torch.nn.functional.pad(psf, (0, shape[-1] - inshape[-1], 0, shape[-2] - inshape[-2], 0, 0))

    # Circularly shift OTF so that the 'center' of the PSF is [0,0] element of the array
    psf = torch.roll(psf, shifts=(-int(inshape[-1] / 2), -int(inshape[-2] / 2)), dims=(-1, -2))

    # Compute the OTF
    otf = fft2(psf)
    return otf


class BlurredBSDS300Dataset(BSDS300Dataset):
    def __init__(self, root='./BSDS300', patch_size=32, split='train', use_patches=True,
                 kernel_size=7, sigma=2):
        super(BlurredBSDS300Dataset, self).__init__(root, patch_size, split, use_patches)

        # trim images to even size
        self.kernel_size = kernel_size
        self.kernel_dataset = MNIST('./', train=True, download=True,
                                    transform=Compose([Lambda(lambda x: np.array(x)),
                                                       ToTensor(),
                                                       Lambda(lambda x: x / torch.sum(x))]))

        kernels = torch.cat([x[0] for (x, _) in zip(self.kernel_dataset, np.arange(self.images.shape[0]))])
        kernels = torch.nn.functional.interpolate(kernels[:, None, ...], size=2*(kernel_size,))
        kernels = kernels / torch.sum(kernels, dim=(-1, -2), keepdim=True)
        self.kernel = kernels[[0]].repeat(kernels.shape[0], 1, 1, 1)

        # blur the images
        H = psf2otf(self.kernel, self.images.shape)
        self.blurred_images = ifft2(fft2(self.images) * H).real
        self.blurred_patches = self.patchify(self.blurred_images, patch_size)

        # save which blur kernel is used for each image
        self.patch_kernel = self.kernel.repeat(1, len(self.blurred_patches) // len(self.images), 1, 1)
        self.patch_kernel = self.patch_kernel.view(-1, *self.kernel.shape[-2:])

        # reshape kernel
        self.kernel = self.kernel.squeeze()

    def get_kernel(self, kernel_size, sigma):
        kernel = self.gaussian(kernel_size, sigma)
        kernel_2d = torch.matmul(kernel.unsqueeze(-1), kernel.unsqueeze(-1).t())
        return kernel_2d

    def __len__(self):
        if self.use_patches:
            return self.blurred_patches.shape[0]
        else:
            return self.blurred_images.shape[0]

    def __getitem__(self, idx):
        if self.use_patches:
            return [self.blurred_patches[idx], self.patches[idx]]
        else:
            return [self.blurred_images[idx], self.images[idx]]


def train(original_model, ckp_path=None, sigma=0.1, epochs=2, batch_size=32, patch_size=32, joint_loss=True):
    print(f'==> Training on noise level {sigma:.02f}')

    # create datasets
    train_dataset = BlurredBSDS300Dataset(patch_size=patch_size, split='train', use_patches=True)

    # create dataloaders & seed for reproducibility
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)


    if original_model:
        model = Unet(norm=None, upsampling_mode='bilinear').to(device)
    else:
        model = ResUnet(norm=None, upsampling_mode='bilinear').to(device)
    optim = torch.optim.Adam(model.parameters(), lr=5e-4)

    epoch_idx = 0

    # load checkpoint
    if ckp_path is not None:
        ckp = torch.load(ckp_path)
        model.load_state_dict(ckp['model_state_dict'])
        optim.load_state_dict(ckp['optimizer_state_dict'])
        epoch_idx = ckp['epoch'] + 1

    losses = []
    psnrs = []
    baseline_psnrs = []
    idx = 0

    # perceptual loss
    layer_indexs = [3, 8, 15]
    perceptual_loss = PerceptualLoss(nn.MSELoss().to(device), layer_indexs, device)

    pbar = tqdm(total=len(train_dataset) * (epochs - epoch_idx) // batch_size)
    while epoch_idx < epochs:
        for blured_sample, gt_sample in train_dataloader:
            model.train()
            gt_sample = gt_sample.to(device)
            blured_sample = blured_sample.to(device)

            # add noise
            noisy_blured_sample = add_noise(blured_sample, sigma=sigma)

            # denoise and deblur
            denoised_deblured_sample = model(noisy_blured_sample)

            # loss function
            if joint_loss:
                lam = 100
                loss = lam * torch.mean((denoised_deblured_sample - gt_sample) ** 2) + \
                       perceptual_loss(denoised_deblured_sample, gt_sample)
            else:
                loss = torch.mean((denoised_deblured_sample - gt_sample)**2)

            psnr = calc_psnr(denoised_deblured_sample, gt_sample)
            baseline_psnr = calc_psnr(noisy_blured_sample, gt_sample)

            losses.append(loss.item())
            psnrs.append(psnr)
            baseline_psnrs.append(baseline_psnr)

            # update model
            optim.zero_grad()
            loss.backward()
            optim.step()

            idx += 1
            pbar.update(1)

        # save checkpoint
        checkpoint = {"model_state_dict": model.state_dict(),
                      "optimizer_state_dict": optim.state_dict(),
                      "epoch": epoch_idx}
        path_checkpoint = "checkpoints/checkpoint_{}_{}_epoch.pkl".format(time.time(), epoch_idx)
        torch.save(checkpoint, path_checkpoint)
        epoch_idx += 1

    pbar.close()
    return model


def evaluate_model(model, sigma=0.1, patch_size=32):
    dataset = BlurredBSDS300Dataset(patch_size=patch_size, split='test', use_patches=False)
    model.eval()

    psnrs = []
    for idx, (blured_image, gt_image) in enumerate(dataset):
        gt_image = gt_image[None, ...].to(device)
        blured_image = blured_image[None, ...].to(device)
        noisy_blured_image = add_noise(blured_image, sigma)
        denoised_deblured_image = model(noisy_blured_image)
        psnr = calc_psnr(denoised_deblured_image, gt_image)
        psnrs.append(psnr)

        if idx == 6:
            skimage.io.imsave('deblur_denoise_out.png', (img_to_numpy(denoised_deblured_image)*255).astype(np.uint8))

    return np.mean(psnrs)


def test(model, sigma=0.1, name_prefix="", patch_size=32):
    dataset = BlurredBSDS300Dataset(patch_size=patch_size, split='test', use_patches=False)
    model.eval()
    for idx, (blured_image, gt_image) in enumerate(dataset):
        blured_image = blured_image[None, ...].to(device)
        noisy_blured_image = add_noise(blured_image, sigma)
        denoised_deblured_image = model(noisy_blured_image)
        skimage.io.imsave(f"deblur_result/{name_prefix}_{sigma}/{name_prefix}_{sigma}_{idx}.png", (img_to_numpy(denoised_deblured_image) * 255).astype(np.uint8))


if __name__ == "__main__":
    model = train(epochs=10, batch_size=64, patch_size=32, joint_loss=True)
    psnr = evaluate_model(model, patch_size=32)
    print(psnr)
