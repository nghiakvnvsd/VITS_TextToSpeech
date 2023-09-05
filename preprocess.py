import argparse
import warnings
import text
from utils import load_filepaths_and_text
from tqdm import tqdm
warnings.filterwarnings("ignore")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--out_extension", default="cleaned")
  parser.add_argument("--text_index", default=2, type=int)
  parser.add_argument("--filelists", nargs="+", default=["../dataset/fonos2_filelist_text.txt"])
  parser.add_argument("--text_cleaners", nargs="+", default=["vietnamese_cleaner"])

  args = parser.parse_args()


  for filelist in args.filelists:
    print("START:", filelist)
    filepaths_and_text = load_filepaths_and_text(filelist)
    for i in tqdm(range(len(filepaths_and_text))):
        try:
            original_text = filepaths_and_text[i][args.text_index]
            cleaned_text = text._clean_text(original_text, args.text_cleaners)
            filepaths_and_text[i][args.text_index] = cleaned_text
        except KeyboardInterrupt:
            pass

    new_filelist = filelist + "." + args.out_extension
    with open(new_filelist, "w", encoding="utf-8") as f:
        f.writelines(["|".join(x) + "\n" for x in filepaths_and_text])
