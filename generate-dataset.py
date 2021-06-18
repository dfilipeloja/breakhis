import pandas as pd
import shutil
import re

df = pd.read_csv('./dataset/Folds.csv')
df = df[(df['fold'] == 1)]
mag = df[(df['mag'] == 200)]

for index, row in mag.iterrows():
    img_name = re.findall(r"[^/]+$", str(row['filename']))[0]

    if row['filename'].startswith('benign/'):
        shutil.copy('dataset/'+row['filename'], 'train/benign/'+img_name)
    elif row['filename'].startswith('malignant/'):
        shutil.copy('dataset/'+row['filename'], 'train/malignant/'+img_name)
