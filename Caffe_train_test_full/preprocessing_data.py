import hashlib
from PIL import Image
import config
import caffe
import numpy as np
import os
import time
import imghdr


def convert_binaryproto_to_npy(binaryproto_file, npy_file='mean.npy'):
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(binaryproto_file, 'rb').read()
    blob.ParseFromString(data)
    arr = np.array(caffe.io.blobproto_to_array(blob))
    np.save(npy_file, arr[0])


def check_corrupt_file(files, type='jpeg', save_file='corrupt.txt',delete=False,print_process=1000):
    list_of_corrupt_files = ''
    for i in range(len(files)):
        if (i % print_process == 0):
            print('Check corrupt: ' + str(i) + " files")
        try:
            img = Image.open(files[i]) # open the image file
            img.verify() # verify that it is, in fact an image
        except (IOError, SyntaxError) as e:
            list_of_corrupt_files += files[i] + '\n'
            if (delete):
                os.remove(files[i])
            print('Corrupt file: '+ files[i])
            continue
        format=imghdr.what(files[i])
        if(format!=type):
            list_of_corrupt_files += files[i] + '\n'
            if (delete):
                os.remove(files[i])
            print('wrong format file: '+ files[i]+', actual format: '+str(format))
    config.save_file(save_file, list_of_corrupt_files)


def get_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

#Function to check duplicate files in dataset
#input:
#   -files: list of file
#   -print_process: to print process
def check_duplicate_files_md5(files, save_file='duplicate.txt', delete=False,print_process=1000):
    files_md5 = []
    is_checked = []
    is_duplicate=[]
    for i in range(len(files)):
        if (i % print_process == 0):
            print('Get md5 of: ' + str(i) + " files")
        files_md5.append(get_md5(files[i]))
        is_checked.append(False)

    for i in range(len(files_md5)):
        if (i % print_process == 0):
            print('Check duplicated: ' + str(i) + " files")
        if (is_checked[i] == True):
            continue
        is_printed = False
        for j in range(i + 1, len(files_md5), 1):
            if (files_md5[j] == files_md5[i]):
                if (is_printed == False):
                    print('Duplicated: ' + str(i) + ' ' + files[i])
                    is_printed = True
                    is_duplicate.append(i)

                is_checked[j] = True
                is_duplicate.append(j)
                print('            ' + str(j) + ' ' + files[j])

        is_checked[i] = True

    list_of_duplicate_files=''
    for i in range(len(is_duplicate)):
        list_of_duplicate_files+=files[is_duplicate[i]]+'\n'
        if(delete):
            os.remove(files[is_duplicate[i]])
    config.save_file(save_file,list_of_duplicate_files)


def main():
    files = config.get_all_files_in_folder(config.data_folder)
    print('Number of files: ' + str(len(files)))
    begin=time.time()
    #files=[]
    #files.append('/home/prdcv/Desktop/zaloAIchallenge/TrainVal/train/91/101810.jpg')
    check_corrupt_file(files,delete=True,save_file=config.data_folder+'/corrupt.txt')
    #check_duplicate_files_md5(files,delete=True,save_file=config.data_folder+'/duplicate.txt')
    end=time.time()
    print('Processing time: '+str(end-begin) +" seconds")
    print('End.')

if __name__ == '__main__':
    main()




