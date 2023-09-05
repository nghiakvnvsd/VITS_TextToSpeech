import sys
import os
""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.
'''
if len(sys.argv) < 2:
    print("Missing argument")
    os.exit(1)
filelist_path = sys.argv[1]

_pad        = '_'
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
_extra = " ̪̪̪"
_extra2 = '̃'
_extra3 = '̃'

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa) + list(_extra) + list(_extra2) + list(_extra3)

symbols = "".join(symbols)
with open(filelist_path, "r") as f:
    data = f.readlines()

data = [line.strip().split("|")[-1] for line in data]
extra = []
for line in data:
    for c in line:
        if c not in symbols:
            extra.append(c)

print(extra)
with open("./extra.txt", "w") as f:
    f.write("".join(extra))
