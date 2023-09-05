import pandas as pd
from tqdm import tqdm

df = pd.read_csv("./csv/filtered_all.csv")
with open("../voice-cloning/filelists/my_train_filelist.txt.cleaned", "r") as f:
    data = f.readlines()

filter_list = set(df["path"])
print(len(filter_list))
with open("../voice-cloning/filelists/my_train_filelist_filtered.txt.cleaned", "w") as f:
    for line in tqdm(data):
        if line.split("|")[0] in filter_list:
            continue
        f.write(line)
