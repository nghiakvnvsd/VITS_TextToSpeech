import os
from tqdm import tqdm

ROOT_DIR = "./dataset"
target = "JVS"
data_dir = os.path.join(ROOT_DIR, target)
target_dir = os.path.join("./dataset16", target)

if not os.path.isdir(target_dir):
    os.makedirs(target_dir)

s_list = os.listdir(data_dir)
s_list = [d for d in s_list if os.path.isdir(os.path.join(data_dir, d))]
for d in tqdm(s_list):
    s_path = os.path.join(data_dir, d)
    s_target_path = os.path.join(target_dir, d)
    if not os.path.isdir(s_target_path):
        os.makedirs(s_target_path)

    filelist = os.listdir(s_path)
    filelist = [file for file in filelist if ".wav" in file]
    for file in filelist:
        file_path = os.path.join(s_path, file)
        out_path = os.path.join(s_target_path, file)
        os.system(f"ffmpeg -i {file_path} -ar 16000 {out_path}")


# filelist = os.listdir(data_dir)
# filelist = [file for file in filelist if ".wav" in file]
# for file in tqdm(filelist):
#     file_path = os.path.join(data_dir, file)
#     out_path = os.path.join(target_dir, file)
#     os.system(f"ffmpeg -i {file_path} -ar 16000 {out_path}")
