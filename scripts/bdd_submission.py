
import os
import json
import shutil
import copy
from tqdm import tqdm


p1 = 'your_experiment_folder/mot_files'  # TODO: your experiment folder
p2 = ''  # TODO: target directory
p3 = 'your_data_path/BDD/test'  # TODO: path to the BDD test or validation set

os.makedirs(p2, exist_ok=True)

for p in tqdm(os.listdir(p1)):
    seq = p[:-5]

    if seq not in os.listdir(p3):
        continue

    with open(os.path.join(p1, p), 'r') as f:
        data = json.load(f)

    num_frames = len(os.listdir(os.path.join(p3, seq, 'img1')))

    for d in data:
        d['videoName'] = seq
        d['frameIndex'] = int(d['name'].split('.')[0].split('-')[-1]) - 1

    empty = {'videoName': seq, 'labels': []}
    
    new_format = list()
    i = 0

    for frame in range(num_frames):
        if data[i]['frameIndex'] == frame:
            new_format.append(data[i])
            i += 1
        else:
            add = copy.deepcopy(empty)
            add['frameIndex'] = frame
            add['name'] = add['videoName'] + '-' + f"{frame+1:07d}.jpg"
            new_format.append(add)

    indices = [d['frameIndex'] for d in new_format]

    sum = 0
    for i, j in zip(indices, indices[1:]):
        sum += j-i
    
    print(num_frames, len(new_format))
    
    assert len(indices) - 1 == sum, f"{seq}, {sum}, {len(indices)}, {indices}"
    assert num_frames == len(new_format), f"{seq}, {sum}, {len(indices)}, {indices}"

    with open(os.path.join(p2, p), 'w') as f:
        json.dump(new_format, f)