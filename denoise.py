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
from skimage.metrics import structural_similarity as ssim

# set random seeds
torch.manual_seed(1)
torch.use_deterministic_algorithms(True)

matplotlib.rcParams['figure.raise_window'] = False

device = torch.device("mps")


class BSDS300Dataset(Dataset):
    def __init__(self, root='./BSDS300', patch_size=32, split='train', use_patches=True):
        files = sorted(glob(os.path.join(root, 'images', split, '*')))

        self.use_patches = use_patches
        self.images = self.load_images(files)
        self.images = self.images[..., :-1, :-1]
        self.patches = self.patchify(self.images, patch_size)
        self.mean = torch.mean(self.patches)
        self.std = torch.std(self.patches)

    def load_images(self, files):
        out = []
        for fname in files:
            img = skimage.io.imread(fname)
            if img.shape[0] > img.shape[1]:
                img = img.transpose(1, 0, 2)
            img = img.transpose(2, 0, 1).astype(np.float32) / 255.
            out.append(torch.from_numpy(img))
        return torch.stack(out)

    def patchify(self, img_array, patch_size):
        # create patches from image array of size (N_images, 3, rows, cols)
        patches = img_array.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.reshape(patches.shape[0], 3, -1, patch_size, patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).reshape(-1, 3, patch_size, patch_size)
        return patches

    def __len__(self):
        if self.use_patches:
            return self.patches.shape[0]
        else:
            return self.images.shape[0]

    def __getitem__(self, idx):
        if self.use_patches:
            return self.patches[idx]
        else:
            return self.images[idx]


def add_noise(x, sigma=0.1):
    return x + torch.randn_like(x) * sigma


def img_to_numpy(x):
    return np.clip(x.detach().cpu().numpy().squeeze().transpose(1, 2, 0), 0, 1)


def calc_psnr(x, gt):
    out = 10 * np.log10(1 / ((x - gt)**2).mean().item())
    return out


def get_feature_module(layer_index, device=None):
    vgg = vgg16(pretrained=True, progress=True).features
    vgg.eval()

    for parm in vgg.parameters():
        parm.requires_grad = False

    feature_module = vgg[0: layer_index + 1]
    feature_module.to(device)
    return feature_module


class PerceptualLoss(nn.Module):
    def __init__(self, loss_func, layer_indexs=None, device=None):
        super(PerceptualLoss, self).__init__()
        self.creation = loss_func
        self.layer_indexs = layer_indexs
        self.device = device

    def forward(self, y, y_):
        loss = 0
        for index in self.layer_indexs:
            feature_module = get_feature_module(index, self.device)
            loss += vgg16_loss(feature_module, self.creation, y, y_)
        return loss


def vgg16_loss(feature_module, loss_func, y, y_):
    out = feature_module(y)
    out_ = feature_module(y_)
    loss = loss_func(out, out_)
    return loss


def train(original_model, ckp_path=None, sigma=0.1, epochs=2, batch_size=32, patch_size=32, joint_loss=True):
    print(f'==> Training on noise level {sigma:.02f}')

    # create datasets
    train_dataset = BSDS300Dataset(patch_size=patch_size, split='train', use_patches=True)

    # create dataloaders & seed for reproducibility
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    if original_model:
        model = Unet(norm=None, upsampling_mode='bilinear').to(device)
    else:
        model = ResUnet(norm=None, upsampling_mode='bilinear').to(device)

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

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
        for sample in train_dataloader:
            model.train()
            sample = sample.to(device)

            # add noise
            noisy_sample = add_noise(sample, sigma=sigma)

            # denoise
            denoised_sample = model(noisy_sample)

            # loss function
            if joint_loss:
                lam = 100
                loss = lam * torch.mean((denoised_sample - sample) ** 2) + perceptual_loss(denoised_sample, sample)
            else:
                loss = torch.mean((denoised_sample - sample)**2)


            psnr = calc_psnr(denoised_sample, sample)
            baseline_psnr = calc_psnr(noisy_sample, sample)

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
    dataset = BSDS300Dataset(patch_size=patch_size, split='test', use_patches=False)
    model.eval()

    psnrs = []
    for idx, image in enumerate(dataset):
        image = image[None, ...].to(device)  # add batch dimension
        noisy_image = add_noise(image, sigma)
        denoised_image = model(noisy_image)
        psnr = calc_psnr(denoised_image, image)
        psnrs.append(psnr)

        if idx == 6:
            skimage.io.imsave('out.png', (img_to_numpy(denoised_image)*255).astype(np.uint8))

    return np.mean(psnrs)


def test(model, sigma=0.1, name_prefix="", patch_size=32):
    dataset = BSDS300Dataset(patch_size=patch_size, split='test', use_patches=False)
    model.eval()

    psnrs = []
    for idx, image in enumerate(dataset):
        image = image[None, ...].to(device)  # add batch dimension
        noisy_image = add_noise(image, sigma)
        denoised_image = model(noisy_image)
        psnr = calc_psnr(denoised_image, image)
        psnrs.append(psnr)
        skimage.io.imsave(f"results/{name_prefix}_{sigma}/{name_prefix}_{sigma}_{idx}.png", (img_to_numpy(denoised_image) * 255).astype(np.uint8))

    return np.mean(psnrs)


def load_pretrained(ckp_path, original_model):
    if original_model:
        model = Unet(norm=None, upsampling_mode='bilinear').to(device)
    else:
        model = ResUnet(norm=None, upsampling_mode='bilinear').to(device)
    ckp = torch.load(ckp_path)
    model.load_state_dict(ckp['model_state_dict'])
    return model


if __name__ == "__main__":
    # train the model
    model = train(original_model=False, epochs=15, batch_size=64, patch_size=32, joint_loss=True, sigma=0.1)
    psnr = evaluate_model(model, patch_size=32, sigma=0.1)
    print(psnr)

    # load the pretained model
    # model = load_pretrained(ckp_path="pretrained_models/resunet_res2_denoise_pretrained.pkl", original_model=False)
