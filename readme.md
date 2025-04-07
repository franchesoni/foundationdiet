# steps
- extract features using DINOv2reg4base from the food101 image dataset, using 8 geometric augmentations (we can divide the data in parts and process each with a different gpu)
- train a supervised baseline over the train set
    (batch_size=192, lr=0.1, epochs=10) `Test Accuracy: 0.906532` 
- train DIET over the train set (without labels)
    (batch_size=192, lr=0.001, epochs=10) Epoch 10/10, Loss: 9.578556, Loss=9.577736
- train the same supervised but over DIET features over the train set
    (batch_size=192, lr=0.1, epochs=10) `Test Accuracy: 0.005032`  
- performance was destroyed, maybe train for longer? train DIET over the train set (without labels) else the indices might be incorrect or the network too small or dinov2reg too good (if it fails still we should try with more augmentations, or trying to get a projector over random nn features, or simply finetune the whole thing and compare it with supervised learning)
    launched (batch_size=192, lr=0.01, epochs=1000) 
    `Test Accuracy: 0.661532` <- worked better but didn't work 



   