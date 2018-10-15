# GrowingGAN
A pyTorch implementation of a progressively grown GAN that learns to generate images from 4x4 to 1024x1024

Requires pyTorch.
Uses https://github.com/facebookresearch/visdom for visualization of training data

GrowingGANv3.py has lots of options to try out different configurations (ConvTranspose instead of Upsample+Convolution, TanH istead of ReLus,
GroupNorm or BatchNorm. 

If the dataset has classes it automatically exploits them.

Trained on a collection of Giger's drawings
![alt text](https://raw.githubusercontent.com/davidebaltieri31/GrowingGAN/master/fake_sample_all_step_118292.png "Giger after 14 epochs")

Trained on MNIST after just 6 epochs
![alt text](https://raw.githubusercontent.com/davidebaltieri31/GrowingGAN/master/GrowingGAN%20on%20mnist_png_training%2024.jpg "MNIST after 24 epochs")

Trained on MNIST after just 6 epochs
![alt text](https://raw.githubusercontent.com/davidebaltieri31/GrowingGAN/master/GrowingGAN%20on%20mnist_png_training.jpg "MNIST after 6 epochs")
