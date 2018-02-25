This folder contains the model for art generation by Neural style transfer.
There are two examples with a cat in Picasso adn Dali styles

<table width="100%">
    <tr>
        <td><img src="images/cat_small.jpg" style="width:300px;height:400px;"></td>
        <td><p align="center"><img src="images/picasso_small.jpg" style="width:300px;height:400px;"></td>
        <td align="right"><img src="output/generated_image.jpg" style="width:300px;height:400px;"></td>
    </tr>

Example with Picasso

<table width="100%">
    <tr>
        <td><img src="images/cat_small.jpg" style="width:300px;height:400px;"></td>
        <td><p align="center"><img src="images/dali_small.jpg" style="width:300px;height:400px;"></td>
        <td align="right"><img src="output_dali/generated_image.jpg" style="width:300px;height:400px;"></td>
    </tr>

And example with Dali

In order to use this model you need to download the pretrained model imagenet-vgg-verydeep-19.mat from http://www.vlfeat.org/matconvnet/pretrained/
Then you can run model.py to generate images presented in output folders.
If you want to generate your image then you need to choose content and style image and put them in images folder.
