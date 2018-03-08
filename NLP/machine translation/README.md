## Generative Adversarial Nets Models

This folder contains Vanilla GAN and DCGAN for fashion generation. 

I would recommend, for those who are not familiar with a GAN concept, to watch [NIPS 2016 Workshop on Adversarial Training](https://www.youtube.com/playlist?list=PLJscN9YDD1buxCitmej1pjJkR5PMhenTF) or to read this [blogpost](https://towardsdatascience.com/understanding-generative-adversarial-networks-4dafc963f2ef) to get a nice overview of GAN model.

### Manual
If you want to generate images then first you need to choose input images for discriminator and set the PATH in the model.py to the folder on your laptop.
Thereafter you can just run the model.py and observe generated images in the out folder

### Models
I was wondering can I generate a fancy t-shirt using latest trends. Therefore, I decided to use GAN in this mission to be ready for the next summer season.
#### Attention
I have used around 100 nice t-shirts for the discriminator, Vanilla GAN and 57k epochs to get this result:
<img src="output/attention.gif">
