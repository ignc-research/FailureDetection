import tempfile
from smb.SMBConnection import SMBConnection

import os

#SMBConnection parameters: <username, pwd, client_name, server_name, use_ntlm_v2 = True>
conn = SMBConnection('mytest', 'mytest', 'jetson-desktop', 'Legion', use_ntlm_v2 = True)
assert conn.connect('192.168.0.10', 139) # sharing server ip address

filelist = conn.listPath('myshare', '/')
local_dir = '/home/jetson/xavier_page/target_folder/'

f_names = []
index = 1
for i in range(len(filelist)):
    if filelist[i].filename.endswith('.jpg'):
        filename = filelist[i].filename
        f_names.append(filename)
        with open(local_dir + filename, 'wb') as fp:
            print(index, '.....copy:', filename)
            index += 1
            conn.retrieveFile('myshare','/'+ filename, fp )
print(len(f_names), 'files copied from share to /home/jetson/xavier_page/target_folder')
#print('found files from smb shared folder', f_names)
