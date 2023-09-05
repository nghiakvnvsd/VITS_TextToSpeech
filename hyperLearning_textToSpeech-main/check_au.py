import os
import pandas as pd
import librosa
from tqdm import tqdm


with open("./filelists/my_train_filelist.txt.cleaned", "r") as f:
    data = f.readlines()

data = [line.strip().split("|")[0] for line in data]
result = []
for path in tqdm(data):
    full_path = os.path.join("../data/dataset16", path)
    au, sr = librosa.load(full_path)
    result.append([path, len(au)])

df = pd.DataFrame(result, columns=["path", "len"])
df.to_csv("../data/csv/check_all.csv", index=False)
