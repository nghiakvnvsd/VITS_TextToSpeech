import librosa
import os
import soundfile
from tqdm import tqdm
import sys

extension = "wav"
if len(sys.argv) < 4:
    print("Missing argument")
    os.exit(1)

# ROOT_DIR = "../../dataset"
ROOT_DIR = sys.argv[1]
TARGET_ROOT_DIR = sys.argv[2]
filelist_path = sys.argv[3]
extension = sys.argv[4]

with open(filelist_path, "r") as f:
    raw_data = f.readlines()
data = [line.strip().split("|")[0] for line in raw_data]

if not os.path.isdir(TARGET_ROOT_DIR):
    os.makedirs(TARGET_ROOT_DIR)

result_data = []
for i, file in tqdm(enumerate(data)):
    file_path = os.path.join(ROOT_DIR, file)
    au, sr = librosa.load(file_path)
    clip, _ = librosa.effects.trim(au, top_db=20)
    target_name = file.replace(f".{extension}", ".wav")
    soundfile.write(os.path.join(TARGET_ROOT_DIR, target_name),
                    clip, sr)
    line = raw_data[i].trim().split("|")
    result_data.append("|".join([target_name, line[1], line[2]]))

with open("trim_" + filelist_path, "w") as f:
    for line in result_data:
        f.write(line + "\n")
