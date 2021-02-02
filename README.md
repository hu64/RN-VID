# RN-VID
Repository for the paper RN-VID: A Feature Fusion Architecture for Video Object Detection
<br> by Hughes Perreault<sup>1</sup>, Pierre Gravel<sup>2</sup>, Maguelonne Héritier<sup>2</sup>, Guillaume-Alexandre Bilodeau<sup>1</sup> and Nicolas Saunier<sup>1</sup>.
<br>
<sup>1</sup> Polytechnique Montréal
<sup>2</sup> Genetec <br>
Paper: https://arxiv.org/abs/2003.10898

## Abstract
Consecutive frames in a video are highly redundant. Therefore, to perform the task of video object detection, executing single frame detectors on every frame without reusing any information is quite wasteful. It is with this idea in mind that we propose RN-VID (standing for RetinaNet-VIDeo), a novel approach to video object detection. Our contributions are twofold. First, we propose a new architecture that allows the usage of information from nearby frames to enhance feature maps. Second, we propose a novel module to merge feature maps of same dimensions using re-ordering of channels and 1 x 1 convolutions. We then demonstrate that RN-VID achieves better mean average precision (mAP) than corresponding single frame detectors with little additional cost during inference.

## Model
![Architecture](images/architecture.png "")

Each frame is passed througha pre-trained VGG-16, and the outputs of block 3, block 4 and block 5 are collected for fusion. B1 to B5 are the standard VGG-16 blocks, and P3 to P7 are the feature pyramid levels. In the dotted frame is an overview of our baseline, a RetinaNet with VGG-16 as a backbone.

<p align="center">
  <img src="https://github.com/hu64/RN-VID/blob/master/fusion_module.png?raw=true" alt="The Fusion Module"/>
</p>
Our fusion module consists of channel re-ordering, concatenation,1×1 convolution, and a final concatenation.

## Award
![Award](images/iciar_award.png "")

This paper was awarded the best paper award at the ICIAR 2020 conference. 

## Acknowledgements
The code for this paper is mainly built upon [keras-retinanet](https://github.com/fizyr/keras-retinanet), we would therefore like to thank the authors for providing their source code. We also acknowledge the support of the Natural Sciences and Engineering Research Council of Canada (NSERC), [RDCPJ 508883 - 17], and the support of Genetec.

## License

RN-VID is released under the MIT License. Portions of the code are borrowed from [keras-retinanet](https://github.com/fizyr/keras-retinanet). Please refer to the original License of this project.
