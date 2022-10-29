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
        counts[i] = num_sprites # number of objects in the scene
        sprite_objects = [0,0,0]

        cat_choice = [0,1,2]
        for j in range(num_sprites):
            pos_y = random_integers(0, height - 12)
            pos_x = random_integers(0, width - 12)

            scale = random_integers(12, min(16, height-pos_y, width-pos_x))

            cat = np.random.choice(cat_choice)
            #cat = random_integers(0,2)
            #cat_choice.remove(cat)
            sprite = np.zeros((height, width, 3))

            # decidet the color of the current paining
            color_channel = random_integers(0,2)

            if cat == 0:  # draw circle
                sprite_objects[0] += 1
                center_x = pos_x + scale // 2.0
                center_y = pos_y + scale // 2.0
                for x in range(height):
                    for y in range(width):
                        dist_center_sq = (x - center_x)**2 + (y - center_y)**2
                        if  dist_center_sq < (scale // 2.0)**2:
                            sprite[x][y][color_channel] = 1.0
            elif cat == 1:  # draw square
                sprite_objects[1] += 1
                sprite[pos_x:pos_x + scale, pos_y:pos_y + scale, color_channel] = 1.0
            else:  # draw square turned by 45 degrees
                sprite_objects[2] += 1
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
        curr_info.append({"question": "Is there any circle?", "program": "exist(filter(scene(),Circle))", "program_id": 2, "answer": str(sprite_objects[0] != 0), "image": i})
        curr_info.append({"question": "Is there any cube", "program": "exist(filter(scene(),Cube))", "program_id": 2, "answer": str(sprite_objects[1] != 0), "image": i})
        curr_info.append({"question": "Is there any diamond?", "program": "exist(filter(scene(),Diamond))", "program_id": 2, "answer": str(sprite_objects[2] != 0), "image": i})
        curr_info.append({"question": "How many objects are there", "program": "count(scene())", "program_id": 2, "answer": str(counts[i]), "image": i})
        qa_dataset.append(curr_info)
        #qa_dataset.append({"question": "How many objects are there?", "program": "count(scene())", "program_id": 2, "answer": str(sprite_objects[0] != 0), "image": i})
    images = np.clip(images, 0.0, 1.0)
    for i in range(images.shape[0]):
        plt.imsave("datasets/sprites3/{}/{}_{}.png".format(split,split,i),images[i])
    save_json(qa_dataset,"datasets/sprites3/{}_sprite3_qa.json".format(split))


if __name__ == "__main__":

    make_sprite3_dataset(10,64,64,"train")