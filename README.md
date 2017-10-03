
Guided Saliency
===
Abstract
---
We applied BiGAN structure to synthesize image with prior input image and its guided saliency. At first stage, we took the combination of perceptual loss and reconstruction loss (saliency) into consideration. Then we added the biGAN loss at the next step to make the synthesized image look more realistic.

Datasets
---
Both training and testing were used the data provided by [COCO](http://cocodataset.org/#home).

We randomly select 15000 images for training and 1100 images for testing from COCO dataset.

We have preprocessed the data into python pickle object files. You could download them in the following link: 

Lasagne
---
The model structure is implemented in [Lasagne](https://github.com/Lasagne/Lasagne), which is developed under [Theano](http://deeplearning.net/software/theano/).

Usage
---

To train the model, please type:
```
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX= float32,dnn.enabled=True,cuda.root=[YOUR CUDA ROOT] python2 02-train.py (auto or bigan)
```
To test the model, please type:
```
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX= float32,dnn.enabled=True,cuda.root=[YOUR CUDA ROOT]python2 03-predict.py
```

Download the pretrained salGAN weights: [salGAN Generator](https://imatge.upc.edu/web/sites/default/files/resources/1720/saliency/2017-salgan/gen_modelWeights0090.npz)


References
---

The coding structure is referred to[saliency-SALGAN-2017](https://github.com/acnokego/saliency-salgan-2017)
