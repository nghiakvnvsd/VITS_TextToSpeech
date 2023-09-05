from tqdm import tqdm
# import pykakasi
# kks = pykakasi.kakasi()
import cutlet
import romkan

katsu = cutlet.Cutlet()
# katsu.romaji("カツカレーは美味しい")

def to_hira(text):
    ro = katsu.romaji(text)
    return romkan.to_hiragana(ro)
    # result = kks.convert(text)
    # ans = ""
    # for item in result:
    #     ans += item['hepburn'] + " "
    # return ans.strip()


with open("./jvs_filelist.txt", "r") as f:
    data = f.readlines()

data = [line.strip() for line in data]

results = []
for line in tqdm(data):
    s = line.split("|")
    text = to_hira(s[-1])
    new_s = "|".join([s[0], s[1], text])
    results.append(new_s)


with open("./jvs_filelist_hira.txt", "w") as f:
    for line in results:
        f.write(line + "\n")
