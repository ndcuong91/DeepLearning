PREPARATION:
+ Build caffe successfully
+ Data folder.

RUN
run in order: 01.create_train_val_folder --> 02.create_lmdb --> 03.make_mean_image -->04.train

01.create_train_val_folder: 
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



Fine-Tuning
Fine-Tuning is the process of training specific sections of a network to improve results.

Making Layers Not Learn
To stop a layer from learning further, you can set it's param attributes in your prototxt.

For example:

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

      
