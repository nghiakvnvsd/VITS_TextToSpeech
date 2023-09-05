import os
import random

with open("../filelists/fonos2_vctk_filelist.txt.cleaned", "r") as f:
    data = f.readlines()


random.shuffle(data)

total = len(data)
num_train = int(0.9 * total)

train_data = data[:num_train]
val_data = data[num_train:]

with open("../filelists/train_fonos2_vctk_filelist.txt.cleaned", "w") as f:
    for line in train_data:
        f.write(line)

with open("../filelists/val_fonos2_vctk_filelist.txt.cleaned", "w") as f:
    for line in val_data:
        f.write(line)
