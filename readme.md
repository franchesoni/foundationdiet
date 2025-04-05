# steps
- extract features using DINOv2reg4base from the food101 image dataset, using 8 geometric augmentations (we can divide the data in parts and process each with a different gpu)
- train a supervised baseline over the train set
    (batch_size=192, lr=0.1, epochs=10) `Test Accuracy: 0.906532` 
- train DIET over the train set (without labels)
    (batch_size=192, lr=0.001, epochs=10) Epoch 10/10, Loss: 9.578556, Loss=9.577736
- train the same supervised but over DIET features over the train set
    (batch_size=192, lr=0.1, epochs=10) `Test Accuracy: 0.005032`  


   