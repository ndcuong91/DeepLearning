import os, random
import config

train_ratio=config.train_ratio
val_ratio=1.0-train_ratio
data_folder=config.data_folder
train_folder=data_folder+'/train'
val_folder=data_folder+'/val'

sub_dir=[os.path.join(data_folder, o) for o in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder,o))]

for i in range(len(sub_dir)):
    sub_dir[i]=sub_dir[i].replace(data_folder+'/','')

if not os.path.exists(train_folder):
    os.makedirs(train_folder)
if not os.path.exists(val_folder):
    os.makedirs(val_folder)

train_txt=''
val_txt=''

for i in range(len(sub_dir)):
    if(sub_dir[i]=='train' or sub_dir[i]=='val' or sub_dir[i]=='train_lmdb'or sub_dir[i]=='val_lmdb'):
        continue
    old_class_folder=data_folder+'/'+sub_dir[i]
    train_class_folder=train_folder+'/'+sub_dir[i]
    val_class_folder=val_folder+'/'+sub_dir[i]
    if not os.path.exists(train_class_folder):
        os.makedirs(train_class_folder)
    if not os.path.exists(val_class_folder):
        os.makedirs(val_class_folder)

    onlyfiles = [f for f in os.listdir(old_class_folder) if os.path.isfile(os.path.join(old_class_folder, f))]
    random.shuffle(onlyfiles)
    num_train_file=int(train_ratio*len(onlyfiles))
    for j in range(len(onlyfiles)):
        if (j<num_train_file):
            os.rename(old_class_folder+'/'+onlyfiles[j], train_class_folder+'/'+onlyfiles[j])
        else:
            os.rename(old_class_folder+'/'+onlyfiles[j], val_class_folder+'/'+onlyfiles[j])

    trainfiles=[f for f in os.listdir(train_class_folder) if os.path.isfile(os.path.join(train_class_folder, f))]
    for k in range(len(trainfiles)):
        train_txt+=sub_dir[i]+'/'+trainfiles[k]+' '+sub_dir[i]+'\n'

    valfiles=[f for f in os.listdir(val_class_folder) if os.path.isfile(os.path.join(val_class_folder, f))]
    for l in range(len(valfiles)):
        val_txt+=sub_dir[i]+'/'+valfiles[l]+' '+sub_dir[i]+'\n'

with open(data_folder+'/train.txt', 'w') as file:
    file.write(train_txt)
with open(data_folder+'/val.txt', 'w') as file:
    file.write(val_txt)
print('end create_train_val')
