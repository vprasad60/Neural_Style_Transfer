# Neural_Style_Transfer
This repository presents an implementation of the Neural Style Transfer (NST) algorithm developed by Leon Gatys, Alexander Ecker, and and Matthias Bethge and described in their [paper](https://arxiv.org/pdf/1508.06576.pdf). Effectively, the algorithm will recreate the content of one image in the style of another. This implementation served as the final project for course in Deep Learning.

## Code Overview and Demonstration
### Designating Images
To run this code, add a content image to the Content_Images folder and a style image to the Style_Images folder. 

### Running the Code
The code snippet below indicates the parameters that can be changed in the notebook file. These are the default parameters used to generate the images located in the 'Generated_Images' folder. It is important to change the content and style image strings based on which images are desired for NST. 

```python
# Establish variables
iters = 25
alpha = 10
beta = 1e4
img_size = 512
model = model_selection()
content_img = 'neckarfront'
style_img = 'haystacks'
input_img = 'content'
content_layers = ['21']
style_layers = ['0','5','10','19','28']
style_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
```

After running the algorithm you will see a progress bar as the iterations are completed. The saved image will appear in the 'Generated_Images' folder.


## References
- leongatys/PytorchNeuralStyleTransfer. (2020). Retrieved 17 November 2020, from https://github.com/leongatys/PytorchNeuralStyleTransfer
- Neural Transfer Using PyTorch — PyTorch Tutorials 1.7.0 documentation. (2020). Retrieved 17 November 2020, from https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
- NeuralStyle Transfer in Pytorch. (2020). Retrieved 17 November 2020, from https://nextjournal.com/gkoehler/pytorch-neural-style-transfer 
- Artistic Neural Style Transfer with PyTorch. Retrived 17 November 2020, from
https://www.pluralsight.com/guides/artistic-neural-style-transfer-with-pytorch
