import os


logdir = './logs'
filelist = os.listdir(logdir)
filelist = [file for file in filelist if ".pth" in file]
if len(filelist) > 4:
    # extract ids
    ids = []
    for file in filelist:
        if 'G_' in file:
            continue
        ids.append(int(file.split('.')[0].split("_")[-1]))

    ids.sort()

    try:
        for i in range(len(ids) - 2):
            idx = ids[i]
            file1 = f"G_{idx}.pth"
            file2 = f"D_{idx}.pth"
            os.remove(os.path.join(logdir, file1))
            os.remove(os.path.join(logdir, file2))
            print("Removed files ", file1, file2)
    except:
        print("FILE NOT FOUND ERROR")
