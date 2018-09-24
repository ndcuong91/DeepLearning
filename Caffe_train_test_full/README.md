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


      
