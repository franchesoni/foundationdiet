# steps
- extract features using DINOv2reg4base from the food101 image dataset, using 8 geometric augmentations (we can divide the data in parts and process each with a different gpu)
- train a supervised baseline over the train set 
- train DIET over the train set (without labels)
- train the same supervised but over DIET features over the train set