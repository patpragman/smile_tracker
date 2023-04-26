import os
import random

# iterate through the contents of a folder and for each item, randomly select another image
# from the other folder


os.chdir("..")

unbalanced_directory = "happy_and_sad_and_else"
balanced_directory = "balanced_set_2"

smiles = os.listdir(f"{unbalanced_directory}/happy")
sad = os.listdir(f"{unbalanced_directory}/sad")
elses = os.listdir(f"{unbalanced_directory}/else")
random.shuffle(elses)

sorted_by_length = [elses, sad, smiles]

for sad_image in sad:
    else_image_name = elses.pop(0)
    happy_image_name = smiles.pop(0)

    os.system(f'cp {unbalanced_directory}/sad/{sad_image} {balanced_directory}/sad/{sad_image}')
    os.system(f'cp {unbalanced_directory}/happy/{happy_image_name} {balanced_directory}/happy/{happy_image_name}')
    os.system(f'cp {unbalanced_directory}/else/{else_image_name} {balanced_directory}/else/{else_image_name}')
