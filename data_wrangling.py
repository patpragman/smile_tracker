import pandas as pd
from os import walk


def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    return(res)

data = {"filepath": [],
        "annotation": []}
for path, _, files in walk("images"):

    for file in files:
        mood = path.split("/")[-1]
        data['filepath'].append(f"{path}/{file}")
        data['annotation'].append(mood)


df = pd.DataFrame(data)
#df = encode_and_bind(df, "annotation")
df.to_csv("annotations.csv",
          index=False)


print(df.iloc[32, 0], df.iloc[32, 1])