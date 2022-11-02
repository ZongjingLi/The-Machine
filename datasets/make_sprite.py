import numpy as np
from numpy.random import random_integers
import matplotlib.pyplot as plt
from moic.utils import save_json,progress_bar

def make_sprite3_dataset(n=60, height=64, width=64,split = "train"):
    images = np.zeros((n, height, width, 3))
    counts = np.zeros((n,))
    qa_dataset = []
    print('Generating sprite dataset...')
    for i in range(n):
        num_sprites = 3 # generate 3 sprite objects

        cat_choice = [0,1,2]

        color_info = {"red":0,"blue":0,"green":0}
        category_info = {"cube":0,"circle":0,"diamond":0}
        for j in range(num_sprites):
            stats = []
            # the location of the object
            pos_y = random_integers(0, height - 12)
            pos_x = random_integers(0, width - 12)

            scale = random_integers(6, min(9, height-pos_y, width-pos_x))

            cat = np.random.choice(cat_choice)
            sprite = np.zeros((height, width, 3))

            # decide the color of the current paining
            color_channel = random_integers(0,2)
            if color_channel == 0:color_info["red"]+=1
            if color_channel == 1:color_info["green"]+=1
            if color_channel == 2:color_info["blue"]+=1

            if cat == 0:  # draw circle
                category_info["circle"] += 1
                center_x = pos_x + scale // 2.0
                center_y = pos_y + scale // 2.0
                for x in range(height):
                    for y in range(width):
                        dist_center_sq = (x - center_x)**2 + (y - center_y)**2
                        if  dist_center_sq < (scale // 2.0)**2:
                            sprite[x][y][color_channel] = 1.0
            elif cat == 1:  # draw square
                category_info["cube"] += 1
                sprite[pos_x:pos_x + scale, pos_y:pos_y + scale, color_channel] = 1.0
            else:  # draw square turned by 45 degrees
                category_info["diamond"] += 1
                center_x = pos_x + scale // 2.0
                center_y = pos_y + scale // 2.0
                for x in range(height):
                    for y in range(width):
                        if abs(x - center_x) + abs(y - center_y) < (scale // 2.0):
                            sprite[x][y][color_channel] = 1.0
            images[i] += sprite
        if i % 100 == 0:
            progress_bar(i, n)
        curr_info = []
        # add two questions about category: exist(filter(scene(),cube))
        category_choose = np.random.choice(["cube","circle","diamond"])
        curr_info.append({"question": "Is there any {}?".format(category_choose), "program": "exist(filter(scene(),{}))".format(category_choose), "program_id": 2, "answer": str(category_info[category_choose] != 0), "image": i})
        category_choose = np.random.choice(["cube","circle","diamond"])
        curr_info.append({"question": "Is there any {}?".format(category_choose), "program": "exist(filter(scene(),{}))".format(category_choose), "program_id": 2, "answer": str(category_info[category_choose] != 0), "image": i})
        # add two questions about color: exist(filter(scene(),red))
        color_choose = np.random.choice(["red","green","blue"])
        curr_info.append({"question": "Is there any {} object?".format(color_choose), "program": "exist(filter(scene(),{}))".format(color_choose), "program_id": 2, "answer": str(color_info[color_choose] != 0), "image": i})
        color_choose = np.random.choice(["red","green","blue"])
        curr_info.append({"question": "Is there any {} object?".format(color_choose), "program": "exist(filter(scene(),{}))".format(color_choose), "program_id": 2, "answer": str(color_info[color_choose] != 0), "image": i})
        # add two question about counting: count(filter(scene(),concept))      
        color_choose = np.random.choice(["red","green","blue"])
        curr_info.append({"question": "how many {} object are there?".format(color_choose), "program": "count(filter(scene(),{}))".format(color_choose), "program_id": 2, "answer": str(color_info[color_choose]), "image": i}) 
        category_choose = np.random.choice(["cube","circle","diamond"])
        curr_info.append({"question": "how many {} are there?".format(category_choose), "program": "count(filter(scene(),{}))".format(category_choose), "program_id": 2, "answer": str(category_info[category_choose]), "image": i}) 
        qa_dataset.append(curr_info)
        #qa_dataset.append({"question": "How many objects are there?", "program": "count(scene())", "program_id": 2, "answer": str(sprite_objects[0] != 0), "image": i})
    images = np.clip(images, 0.0, 1.0)
    for i in range(images.shape[0]):
        plt.imsave("datasets/sprites3/{}/{}_{}.png".format(split,split,i),images[i])
    save_json(qa_dataset,"datasets/sprites3/{}_sprite3_qa.json".format(split))


if __name__ == "__main__":

    make_sprite3_dataset(3600,32,32,"train")