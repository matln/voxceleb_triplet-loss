# voxceleb_triplet-loss
A Pytorch implementation of triplet loss on VoxCeleb1

# Train
Using softmax pre-training to initialise the network.

triplets selector: semi-hard 

# Test
Full length utterances are used in testing. So we can fed into network just one example everytime. But if using multi-GPUs(e.g. 6), you can input 6 examples for each iteration.

# Result 
embedding_dim=1024:

-|EER|m-DCF
:--|:--|:--
pretrain|8.62|0.70
triplet loss|7.57|0.68

# Reference
https://github.com/davidsandberg/facenet
https://github.com/adambielski/siamese-triplet
https://github.com/v-iashin/VoxCeleb

