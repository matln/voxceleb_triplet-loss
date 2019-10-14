# voxceleb_triplet-loss
A Pytorch implementation of triplet loss on VoxCeleb1

# Train
Using softmax pre-training to initialise the network.

triplets selector: semi-hard 

# Test
Full length utterances are used in testing. So we can fed into network just one example everytime. But if using multi-GPUs(e.g. 6), you can input 6 examples for each iteration.

# Result 
embedding_dim=256:

-|EER|m-DCF
:--|:--|:--
triplet loss|7.22|0.639

# Usage

* Create virtual environment:
  ```python
  conda create -n vox_triplet_loss python=3.7
  conda activate vox_triplet_loss
  ```
  
* Install `pytorch`:
  https://pytorch.org/get-started/locally/
  
* Pretrain:
  * Change the `train_mode` to `‘pretrain’` in `constants.py`
  * Set GPU configuration, data_path and save_path in your computer
  * Run the `train.py`
  
* Use triplet loss to train model:
  * Change the `train_mode` to `‘triplet’` in `constants.py`

# Reference
https://github.com/davidsandberg/facenet

https://github.com/adambielski/siamese-triplet

https://github.com/v-iashin/VoxCeleb

