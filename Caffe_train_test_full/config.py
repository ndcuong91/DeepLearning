import os
#create_train_val configs
data_folder='/home/prdcv/Desktop/zaloAIchallenge/TrainVal'
train_ratio=0.92

def save_file(file_path, content):
    with open(file_path, 'w') as file:
        file.write(content)

def get_all_files_in_folder(folder):
    all_files=[]
    for path, subdirs, files in os.walk(folder):
        for name in files:
            all_files.append(os.path.join(path, name))
    return all_files


