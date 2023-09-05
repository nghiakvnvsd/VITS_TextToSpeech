from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForXVector
import librosa
import torch
import os
import sys
from tqdm import tqdm

if len(sys.argv) < 3:
    print("Missing argument")
    os.exit(1)

# ROOT_DIR = "../../dataset"
ROOT_DIR = sys.argv[1]
filelist_path = sys.argv[2]

device = "cpu"
batch_size = 32
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("anton-l/wav2vec2-base-superb-sv")
model = Wav2Vec2ForXVector.from_pretrained("anton-l/wav2vec2-base-superb-sv").cpu()
with open(filelist_path, "r") as f:
    data = f.readlines()

data = [line.strip().split("|")[0] for line in data]
cnt = 0
batch = []
names = []
for idx in tqdm(range(0, len(data))):
  line = data[idx]
  file_path = os.path.join(ROOT_DIR, line)
  embed_path = file_path.replace(".wav", ".emb.pt")
  cnt += 1
  au, sr = librosa.load(file_path, sr=16000)
  batch.append(au)
  names.append(embed_path)

  if cnt >= batch_size or idx == len(data)-1:
    cnt = 0
    inputs = feature_extractor(
      batch, sampling_rate=sr, return_tensors="pt", padding=True
    )
    with torch.no_grad():
      embeddings = model(input_values=inputs["input_values"].to(device),
                         attention_mask=inputs["attention_mask"].to(device)).embeddings

    for n, t in zip(names, embeddings):
      torch.save(t, n)
    batch = []
    names = []
    del embeddings, inputs
