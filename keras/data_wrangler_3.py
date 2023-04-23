import os
import random

# iterate through the contents of a folder and for each item, randomly select another image
# from the other folder


os.chdir("..")

unbalanced_directory = "smiles_and_else"
balanced_directory = "balanced"

smiles = os.listdir(f"{unbalanced_directory}/happy")
elses = os.listdir(f"{unbalanced_directory}/else")
random.shuffle(elses)

for smile_image_name in smiles:
    else_image_name = elses.pop(0)
    os.system(f'cp {unbalanced_directory}/happy/{smile_image_name} {balanced_directory}/happy/{smile_image_name}')
    os.system(f'cp {unbalanced_directory}/else/{else_image_name} {balanced_directory}/else/{else_image_name}')
