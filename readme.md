# Denoise
The code of train and test of denoising task is in _denoise.py_

# Deblur & Denoise
The code of train and test of deblurring and denoising task is in _deblur_denoise.py_

# Pretrained models
For denoise:
* UNet: _unet_denoise_pretrained.pkl_
* ResUNet: _resunet_res1_denoise_pretrained.pkl_ (one res-block in each downsampling block), _resunet_res2_denoise_pretrained.pkl_ (two res-block in each downsampling block)
* ResUNet with Perceptual Loss: _resunet_pre_denoise_pretrained.pkl_

For deblur & denoise:
* Unet: _unet_deblur_denoise_pretrained.pkl_
* ResUNet: _resunet_deblur_denoise_pretrained.pkl_
* ResUNet with Perceptual Loss: _resunet_per_deblur_denoise_pretrained.pkl_

If you want to use the pretrained model, please use the `load_pretrained` method.
For example: 

`model = load_pretrained(ckp_path="pretrained_models/resunet_res2_denoise_pretrained.pkl", original_model=False)`

Then you use the `evaluate_model` method to evaluate the model.