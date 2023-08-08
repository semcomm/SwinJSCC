# SwinJSCC: Taming Swin Transformer for Joint Source-Channel Coding

This paper has been partially presented in IEEE ICASSP 2023 "[WITT: A Wireless Image Transmission Transformer For Semantic Communications](https://arxiv.org/abs/2211.00937)" and the code is available at [](https://github.com/KeYang8/WITT).

## Introduction
In this paper, we establish a more expressive JSCC codec architecture that can also adapt flexibly to diverse channel states and transmission rates within a single model. Specifically, we demonstrate that with elaborate design, neural JSCC codec built on the emerging Swin Transformer backbone can achieve superior performance than conventional neural JSCC codecs built upon CNN while also requiring lower end-to-end processing latency. Paired with two well-designed spatial modulation modules that scale latent representations based on the channel state information and target transmission rate, our baseline SwinJSCC can further upgrade to a versatile version, which increases its capability to adapt to diverse channel conditions and rate configurations. Extensive experimental results show that our SwinJSCC achieves better or comparable performance versus the state-of-the-art engineered BPG + 5G LDPC coded transmission system with much faster end-to-end coding speed, especially for high-resolution images, in which case traditional CNN-based JSCC yet falls behind due to its limited model capacity. 
![ ](overview.png)
>  The overall architecture of the proposed SwinJSCC scheme for wireless image transmission.


# Acknowledgement
The implementation is based on [Swin Transformer](https://github.com/microsoft/Swin-Transformer).

# Related links
* BPG image format by _Fabrice Bellard_: https://bellard.org/bpg
* Sionna An Open-Source Library for Next-Generation Physical Layer Research: https://github.com/NVlabs/sionna
* DIV2K image dataset: https://data.vision.ee.ethz.ch/cvl/DIV2K/
* Kodak image dataset: http://r0k.us/graphics/kodak/
* CLIC image dataset:  http://compression.cc
