# VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech

## Installation
1. Python packages
```
pip install -r requirements.txt
```

2. Install Pytorch version 1.9.0 for CUDA 11  
```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

3. Cython-version Monotonoic Alignment Search  
```
cd monotonic_align
python setup.py build_ext --inplace
```

## Data Preprocessing 
1. Make a filelist for each dataset to match the format  
```
file name|speaker id|text
```

Example:  
```
FPTOpenSpeech/FPTOpenSpeechData_Set001_V0.1_000001.wav|126|cách để đi
FPTOpenSpeech/FPTOpenSpeechData_Set001_V0.1_000002.wav|349|họ đã xét nghiệm máu cho cheng nhưng mọi thứ vẫn hoàn toàn bình thường
FPTOpenSpeech/FPTOpenSpeechData_Set001_V0.1_000003.wav|217|anh có thể gọi tôi không
FPTOpenSpeech/FPTOpenSpeechData_Set001_V0.1_000004.wav|346|có rất nhiều yếu tố may rủi ở đây
FPTOpenSpeech/FPTOpenSpeechData_Set001_V0.1_000005.wav|142|ai là chúa nói dối
FPTOpenSpeech/FPTOpenSpeechData_Set001_V0.1_000006.wav|195|có cửa hàng tiện lợi ở sân bay không

```

2. Trim audios  
```
python my_data/trim_audio.py [ROOT_DIR] [TARGET_ROOT_DIR] [filelist path] [extension]
```
ROOT_DIR: current root directory of your data  
TARGET_ROOT_DIR: root directory for your trimmed data  
filelist path: path to your filelist.txt
extension: your audio file extension - do not include the dot (the script will convert your audio files to .wav)
Example:
```
python my_data/trim_audio.py "../dataset" "../trim_dataset" "filelist.txt" "mp3"
```

3. Extract speaker embeddings from audios  
```
python my_data/create_embed.py [dataset root directory] [path to your filelist]
```

4. Convert text to IPA
```
python ./preprocess.py --filelists [path to your filelist.txt]
```
After processing, a file named [your filelist name].txt.cleaned will be generated.  


5. Extract extra symbols
```
python my_data/extract_symbol.py [path to your filelist.txt.cleaned]
```
A list of extra symbols will be printed on the screen. Copy these symbols and paste in the text/symbols.py script
```

...
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
_extra = ""  <-- Paste extra symbols here
...

```
## Configuration
Modify the filelist path in the file configs/my_base.json  
```
  "data": {
    "training_files": "[PATH TO YOUR training filelist].txt.cleaned",
    "validation_files": "[PATH TO YOUR validation filelist].txt.cleaned",
	"data_dir": "[YOUR DATASET ROOT DIRECTORY]",
	...
	}
```

## Training
```
python ./train.py -m [experiment name]
```

## Inference
```
python ./test.py [model path] [voice sample path]
```
