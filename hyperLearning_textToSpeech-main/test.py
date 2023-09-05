import soundfile
import sys
from playsound import playsound
import torch

import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path


voice_ref_path = "./dataset/wav/VIVOSSPK38_001.wav"
model_path = "logs/G_150080.pth"
if len(sys.argv) >= 3:
    model_path = sys.argv[1]
    voice_ref_path = sys.argv[2]


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


# voice sample path
fpath = Path(voice_ref_path)
# fpath = Path("../samples/trungitn.wav")
wav = preprocess_wav(fpath)

encoder = VoiceEncoder()
embed_ref = encoder.embed_utterance(wav)
embed_ref = torch.tensor(embed_ref).cpu().unsqueeze(0)

hps = utils.get_hparams_from_file("./configs/my_base.json")
net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).cpu()
net_g.eval()

utils.load_checkpoint(model_path, net_g, None)
# utils.load_checkpoint("logs/old/G_129900.pth", net_g, None)

while True:
    text = input("Type your text here: ")
    if text.lower() == 'q':
        break
    print("Generating...")
    stn_tst = get_text(text, hps)
    with torch.no_grad():
        x_tst = stn_tst.cpu().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cpu()
        # o = net_g.infer(x_tst, x_tst_lengths, embed_ref, noise_scale=.5, noise_scale_w=0.5, length_scale=1)
        audio = net_g.infer(x_tst, x_tst_lengths, embed_ref, noise_scale=.5, noise_scale_w=0.5, length_scale=1)[0][0,0].data.cpu().float().numpy()
    soundfile.write("./inference/test.wav", audio, hps.data.sampling_rate)
    print("Talking...")
    playsound("./inference/test.wav")
    print("Done.")
