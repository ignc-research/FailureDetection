from os import walk
from os import remove
from shutil import copyfile
from threading import Timer
import threading
import time

def do_compare(source_folder_path='source_folder', target_folder_path='target_folder', target='TM'):
    # source_folder_path = 'source_folder'
    # target_folder_path = 'target_folder'
    '''
    parameter 'target' is a Tuple
    '''

    f1, f2 = [], []

    for (dirpath, dirnames, filenames) in walk(source_folder_path):
        for file in filenames:
            if target in file:
                f1.append(file)

    for (dirpath, dirnames, filenames) in walk(target_folder_path):
        f2.extend(filenames)

    for file in f1:
        if file not in f2:
            print(file, 'gefunden and kopiert')
            copyfile(source_folder_path+'/'+file, target_folder_path+'/'+file)


def clear_folder(target_folder=None):
    '''
    clear files in the target_folder
    '''
    for (dirpath, dirnames, filenames) in walk(target_folder):
        for file in filenames:
            remove(target_folder +'/'+ file)


if __name__ == '__main__':
    num = 0
    while True:
        time.sleep(1)
        print(num, "hello world.")
        do_compare()
        #clear_folder('target_folder')
        num += 1

