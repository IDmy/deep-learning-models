## Generative Adversarial Nets Models

This folder contains the GAN models for fashion generation.

The first implemented model is a Vanilla GAN. Later it will be compared with DCGAN.
I would recommend, for those who are not familiar with a GAN concept, to read this [blogpost](https://towardsdatascience.com/understanding-generative-adversarial-networks-4dafc963f2ef) to get a nice overview of GAN model.

### Manual

If you want to generate images then first you need to choose input images for discriminator and set the PATH in the model.py to the folder on your laptop.
Thereafter you can just run the model.py and observe generated images in the out folder

Since I was wondering can I generate a fancy t-shirt using latest trends, I decided to use GAN in this mission to be ready for the next summer season.
I have used around 100 nice t-shirts for the discriminator, Vanilla GAN and 57k epochs to get this result:

#### Vanilla GAN
<img src="out_man_tshirts_4/generated_tshirts.gif"  height="200" width="200">

Not nice so far. Therefore the next step is to use Deep Convolutional GAN (DCGAN), which has convolutional layers and batch normalization that are supposed to help with the stability of the convergence. The original DCGAN uses 4 convolutional layers but I used only 3 to speed up calculations. Even with faster architecture I made only 20k epochs. Therefore, I suppose the result has to be better with 4 layers and more epochs. Also tuning of parameters may help to improve the result.   

#### DCGAN
<img src="dcgan_out_man_tshirts/tshirts.gif" height="150" width="150">

It looks that in both cases the generator learns the shape of t-shirts and in case of DCGAN performs better color generation. All t-shirts in train dataset have different design, color and prints that make this task quite difficult for the GAN.
