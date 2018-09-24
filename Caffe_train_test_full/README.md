# Caffe 1.0 training testing from A to Z on Ubuntu
**This repo is a complete guide to work with Caffe on Ubuntu.**
## Preparation:
+ Build caffe successfully https://github.com/BVLC/caffe/releases/tag/1.0 
+ Data folder (contain images in separate class)

## Run
### 1. Create train and valid folder
+ Use script *01.create_train_val_folder*
+ input:"Data" folder:
  +Data:
    \1:
      \1.jpg
      \2.jpg
      ...
    \2:
      \3.jpg
      \4.jpg
   
+ output: "Data" folder with train/val folders and train.txt, val.txt 
  +Data:
    \train:
      \1:
        \1.jpg
        \2.jpg
        ...
      \2:
        \3.jpg
        \4.jpg
      .....
    \val: 
      \1:
        \1.jpg
        \2.jpg
        ...
      \2:
        \3.jpg
        \4.jpg
      .....
    \train.txt (list all files in train folder)
    \val.txt (list all files in val folder)
### 2. Create LMDB file
+ Caffe will work faster with LMDB data. To use it, run *02.create_lmdb.sh*

### 3. Make mean image
+ The training process in Caffe will converge faster with data's nomarlization technique. Run *03.make_mean_image.sh* to use it

### 4. Training
+ Now you are ready for training, simply use *04.1.train.sh* or *04.2.train_resume.sh* to resume training from snapshot

## Others
### Clean Data
Use preprocessing_data.py to check corrupt or duplicated files on Data folder


### Fine-Tuning
**Fine-Tuning is the process of training specific sections of a network to improve results.**

**To stop a layer from learning further**, you can set it's param attributes in your prototxt.

For example:
```
layer {
  name: "example"
  type: "example" 
  ...
  param {
    lr_mult: 0    #learning rate of weights
    decay_mult: 1
  }
  param {
    lr_mult: 0    #learning rate of bias
    decay_mult: 0
  }
}
```

### Evaluate result
+ Use script *eval_val_test.py* to evaluate your result on valid and test set
      
