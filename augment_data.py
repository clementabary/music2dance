import os
import shutil
import json
import numpy as np
from visualize import frame_to_vid
from tqdm import tqdm


datafolder = './Music-to-Dance-Motion-Synthesis-master'
files_to_be_copied = {'new_skeletons.json', 'skeletons.json'}

for dir in tqdm(os.listdir(datafolder)):
    dir = '{}/{}'.format(datafolder, dir)
    if os.path.isdir(dir) and os.path.basename(dir)[0:5] == 'DANCE' and os.path.basename(dir)[-3:] != 'bis' and os.path.basename(dir)[6] == 'W':
        newdir = dir + 'bis'
        if not os.path.exists(newdir):
            os.mkdir(newdir)
        for file in os.listdir(dir):
            if file in files_to_be_copied:
                shutil.copy(os.path.join(dir, file), newdir)
        if os.path.basename(dir)[6] == 'W':
            skf = '/new_skeletons.json'
        else:
            skf = '/skeletons.json'
        with open(dir + skf) as f:
            data = json.load(f)
            centers = data['center']
            length = data['length']
            skeletons = np.asarray(data['skeletons'])
            fixed_skeletons = skeletons.copy()
            fixed_skeletons[:, :, 0] = - fixed_skeletons[:, :, 0]
            new_data = {'length': length, 'center': centers, 'skeletons': fixed_skeletons.tolist()}
            frame_to_vid(fixed_skeletons, newdir + '/animation.avi', fps=25)
        with open(newdir + skf, 'w') as f2:
            json.dump(new_data, f2)


for dir in tqdm(os.listdir(datafolder)):
    dir = '{}/{}'.format(datafolder, dir)
    if os.path.basename(dir)[0:5] == 'DANCE' and os.path.basename(dir)[-3:] != 'bis' and os.path.basename(dir)[6] == 'W':
        with open(dir + '/fixed_skeletons.json') as f:
            data = json.load(f)
            frame_to_vid(np.asarray(data['skeletons']), dir + '/animation.avi', fps=25)
