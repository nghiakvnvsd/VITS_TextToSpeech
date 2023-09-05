import os

with open("../filelists/my_train_filelist_filtered.txt.cleaned", "r") as f:
    data = f.readlines()


with open("../filelists/vivo_filelist.txt", "w") as f:
    for line in data:
        if "VIVO" in line:
            f.write(line)

