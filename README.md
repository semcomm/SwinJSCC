# SwinJSCC: Taming Swin Transformer for Joint Source-Channel Coding

Official Pytorch implementation for "[SwinJSCC: Taming Swin Transformer for Deep Joint Source-Channel Coding](https://arxiv.org/abs/2308.09361)".

## Introduction

In this paper, we establish a new neural JSCC backbone that can also adapt flexibly to diverse channel conditions and transmission rates within a single model, our open-source project aims to promote the research in this field. Specifically, we show that with elaborate design, neural JSCC codec built on the emerging Swin Transformer backbone achieves superior performance than conventional neural JSCC codecs built upon CNN, while also requiring lower end-to-end processing latency. Paired with two spatial modulation modules that scale latent representations based on the channel state information and target transmission rate, our baseline SwinJSCC can further upgrade to a versatile version, which increases its capability to adapt to diverse channel conditions and rate configurations. Extensive experimental results show that our SwinJSCC achieves better or comparable performance versus the state-of-the-art engineered BPG + 5G LDPC coded transmission system with much faster end-to-end coding speed, especially for high-resolution images, in which case traditional CNN-based JSCC yet falls behind due to its limited model capacity. 

## Installation
We implement SwinJSCC under Python 3.8 and PyTorch 1.9.

Note: We found that when using w/ SA & RA for inference in a torch > 1.12 environment, the results are inconsistent with those reported in the paper. Therefore, it is recommended to use the same version of PyTorch as the training environment (torch <= 1.12) for inference.


## Usage

All pretrained models are in [Google Drive](https://drive.google.com/drive/folders/1_EouRY4yYvMCtamX2ReBzEd5YBQbyesc?usp=sharing).

* cbr = C/(2^(2i)*3*2), i denotes the downsample number. For CIFAR10, i=2; for HR_image, i=4.
* SwinJSCC_w/o_SAandRA model is the SwinJSCC model without Channel ModNet module and Rate ModNet which is trained on a fixed SNR  and rate. SwinJSCC_w/_SA model is the SwinJSCC model with Channel ModNet module which is trained on various SNRs and a fixed rate. SwinJSCC_w/_RA model is the SwinJSCC model with Rate ModNet module which is trained on various rates and a fixed SNR. SwinJSCC_w/_SAandRA model is the SwinJSCC model with Rate ModNet module and Channel ModNet module which is trained on various rates and SNRs.
* 'multiple-snr' decides use either fixed or random SNR to train the model. For models which without Channel ModNet module, 'multiple-snr' is set as a fixed SNR. For models which with Channel ModNet module, 'muliple-snr' can be set as both fixed or random SNRs.
* 'C' decides use either fixed or random rate to train the model. For models which without Rate ModNet module, 'C' is set as a fixed rate. For models which with Rate ModNet module, 'C' can be set as both fixed or random rates.
* 'model_size' decides model params size, we set three model sizes, e.g. small, base, large.
* for high-resolution images, we can firstly train the SwinJSCC_W/O model. Then, the SwinJSCC_W/O model is used as a pre-training model to train the whole SwinJSCC model.
* You can apply our method on your own images.
```
python main.py --training --trainset {CIFAR10/DIV2K} --testset {CIFAR10/kodak/CLIC21} -- distortion-metric {MSE/MS-SSIM} --model {'SwinJSCC_w/o_SAandRA'/'SwinJSCC_w/_SA'/'SwinJSCC_w/_RA'/'SwinJSCC_w/_SAandRA'} --channel-type {awgn/rayleigh} --C {bottleneck dimension} --multiple-snr {random or fixed snr} --model_size {SwinJSCC model size}
```

### For SwinJSCC_w/o_SAandRA model 

*e.g. cbr = 0.0625, snr = 10, metric = PSNR, channel = AWGN

```
e.g.
python main.py --trainset DIV2K --testset kodak -- distortion-metric MSE --model SwinJSCC_w/o_SAandRA model --channel-type awgn --C 96 -- multiple-snr 10 --model_size base
```

You can apply our method on your own images.

### For SwinJSCC_w/_SA model 

*e.g. cbr = 0.0625, snr = 1,4,7,10,13, metric = PSNR, channel = AWGN

```
e.g.
python main.py --trainset DIV2K --testset kodak --distortion-metric MSE --model SwinJSCC_w/_SA --channel-type awgn --C 96 --multiple-snr 1,4,7,10,13 --model_size base
```
### For SwinJSCC_w/_RA model 
*e.g. cbr = 0.0208,0.0416,0.0625,0.0833,0.125, snr = 10, metric = PSNR, channel = AWGN

```
e.g.
python main.py --trainset DIV2K --testset kodak --distortion-metric MSE --model SwinJSCC_w/_RA --channel-type awgn --C 32,64,96,128,192 --multiple-snr 10 --model_size base
```

### For SwinJSCC_w/_SAandRA model 
*e.g. cbr = 0.0208,0.0416,0.0625,0.0833,0.125, snr = 1,4,7,10,13, metric = PSNR, channel = AWGN

```
e.g.
python main.py --trainset DIV2K --testset kodak --distortion-metric MSE --model SwinJSCC_w/_SAandRA --channel-type awgn --C 32,64,96,128,192 --multiple-snr 1,4,7,10,13 --model_size base
```

>If you want to train this model, please add '--training'. 


## Citation

If you find this work useful for your research, please cite:

```
@ARTICLE{10589474,
  author={Yang, Ke and Wang, Sixian and Dai, Jincheng and Qin, Xiaoqi and Niu, Kai and Zhang, Ping},
  journal={IEEE Transactions on Cognitive Communications and Networking}, 
  title={SwinJSCC: Taming Swin Transformer for Deep Joint Source-Channel Coding}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Transformers;Adaptation models;Signal to noise ratio;Convolutional neural networks;Wireless communication;Vectors;Image coding;Joint source-channel coding;Swin Transformer;attention mechanism;image communications},
  doi={10.1109/TCCN.2024.3424842}
}
```

## Acknowledgement
The implementation is based on [Swin Transformer](https://github.com/microsoft/Swin-Transformer).

## Related links
* BPG image format by _Fabrice Bellard_: https://bellard.org/bpg
* Sionna An Open-Source Library for Next-Generation Physical Layer Research: https://github.com/NVlabs/sionna
* DIV2K image dataset: https://data.vision.ee.ethz.ch/cvl/DIV2K/
* Kodak image dataset: http://r0k.us/graphics/kodak/
* CLIC image dataset:  http://compression.cc
