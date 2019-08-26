Pytorch implementation of "Avatar-Net: Multi-scale Zero-shot Style Transfer by Feature Decoration"
---

Pytorch implementation of the paper:
"Avatar-Net: Multi-scale Zero-shot Style Transfer by Feature Decoration"

Lu Sheng, Ziyi Lin, Jing Shao, Xiaogang Wang

CVPR 2018

Dependencies
--
* Pytorch (version >= 0.4.0)

Download
--
* The trained models can be downloaded throuth the [Google drive](https://drive.google.com/drive/folders/1JDgn5oO11AWnbpUxpyrdPe_pYwgfGhSu?usp=sharing).
* [MSCOCO train2014](http://cocodataset.org/#download) is needed to train the network.

Usage
--

### Train example script

```
python main.py --train-flag True --cuda-device-no 0 --imsize 512 --cropsize 256 --train-data-path ./coco2014/ --save-path ./trained_models/
```

### Test example script

```
python main.py --train-flag False --cuda-device-no 0 --imsize 512 --model-load-path trained_models/network.pth --test-content-image-path sample_images/content_images/chicago.jpg --test-style-image-path sample_images/style_images/mondrian.jpg --output-image-path chicago_mondrian.png --style-strength 1.0 --patch-size 3 --patch-stride 1
```

Example Images
--

### Stylize a content image with the target style image.

* Content image: sample_images/content_images/chicago.jpg
* Style image: sample_images/style_images/mondrian.jpg

![test_result](https://github.com/tyui592/Avatar-Net_Pytorch/blob/master/sample_images/test_results/chicago_mondrian.png)

* Content image: smaple_images/content_images/cornell.jpg
* Style image: sample_images/style_images/candy.jpg

![test_result2](https://github.com/tyui592/Avatar-Net_Pytorch/blob/master/sample_images/test_results/cornell_candy.png)

### Style interpolation

* Content image: smaple_images/content_images/chicago.jpg
* Style image: sample_images/style_images/abstraction.jpg

![test_result3](https://github.com/tyui592/Avatar-Net_Pytorch/blob/master/sample_images/test_results/chicago_abstraction_style-interpolation.png)

### Patch size change

* Content image: smaple_images/content_images/chicago.jpg
* Style image: sample_images/style_images/abstraction.jpg

![test_result4](https://github.com/tyui592/Avatar-Net_Pytorch/blob/master/sample_images/test_results/chicago_abstraction_patch-size.png)
