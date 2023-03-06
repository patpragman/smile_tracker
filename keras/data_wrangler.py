import os
import pandas as pd

annotes = pd.read_csv("../annotations.csv")

for i, (file_path, annotation) in annotes.iterrows():
    splits = file_path.split("/")
    category = splits[2]
    filename = splits[-1]
    file_path = f"../{file_path}"

    if not os.path.isdir(f"images/{category}"):
        os.mkdir(f"images/{category}")

    if os.path.isfile(f"images/{category}/{filename}"):
        filename = f"addtional_{filename}"

    os.system(f'cp {file_path} images/{category}/{filename}')

