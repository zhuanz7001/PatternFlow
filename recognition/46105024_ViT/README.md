# Classify Alzheimer’s disease by vision transformer
## Main Purpose
This project aims to implement a vision transformer to Classify Alzheimer’s disease and have a minimum accuracy of 0.8 on the test set. I am sorry that I did not achieve this goal and the code of this project may have some problems. In this document, I will describe the structure of the model and the parts that have been implemented.
## Model structue
The main modules of ViT include main 3 parts: "Embedded Patches", "Muti-Head Attention" and "Feed Forward"
![](images/ViT.png)
(Dosovitskiy et al., 2020)
### Embedded Patches
The first step is split the picture.
The image x ∈ H × W × C, where（H, W） is the resolution of the original image and C is the number of channels. Dividing it into non-overlapping patches of size P × P.</br>
The second step is patch embedding.
Flatten the patches and map to D dimensions, which similar to BERT's word embedding operation.</br>
The Third step is position embedding.
1D position embeddings are added to the patch embeddings to keep the positional information.
### Muti-Head Attention
