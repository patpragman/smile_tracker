import os
import pandas as pd

annotes = pd.read_csv("../annotations.csv")

for i, (file_path, annotation) in annotes.iterrows():
    splits = file_path.split("/")
    category = splits[-2]
    old_fname = splits[-1]
    filename = f"{category}_{splits[-1]}"

    if category.upper() != "HAPPY" and category.upper() != "SAD":
        category = "else"



    if os.path.isfile(f"../happy_and_sad_and_else/{category}/{filename}"):
        filename = f"additional_{filename}"

    os.system(f'cp ../example/{file_path} ../happy_and_sad_and_else/{category}/{filename}')
