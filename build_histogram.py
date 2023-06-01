import json
from pathlib import Path
from math import sqrt
from tqdm import tqdm


data_path = './rgb_voxel_data/train/(smaller noise35*30*50)stack-block-pyramid'
data_path = Path(data_path)
files = data_path.glob('**/*_ROI.json')
output = {}

for _, file in tqdm(enumerate(files), total=2400):
    with open(file, 'r') as data:
        data = json.load(data)
    keys = data.keys()
    
    for i, key in enumerate(keys):
        for key2 in list(keys)[i+1:]:
            x_dist = []
            y_dist = []
            z_dist = []
            xy_dist = []
            yz_dist = []
            xz_dist = []
            xyz_dist = []
            for i in range(101):
                x_dist.append(0)
                y_dist.append(0)
                z_dist.append(0)
                xy_dist.append(0)
                yz_dist.append(0)
                xz_dist.append(0)
                xyz_dist.append(0)
            d10 = data[key]['front_bottom_left']
            d11 = data[key]['back_top_right']
            d20 = data[key2]['front_bottom_left']
            d21 = data[key2]['back_top_right']
            voxels = 0
            for x in [d10[0], d11[0]]:
                for y in [d10[1], d11[1]]:
                    for z in [d10[2], d11[2]]:
                        for x2 in [d20[0], d21[0]]:
                            for y2 in[d20[1], d21[1]]:
                                for z2 in [d20[2], d21[2]]:
                                    voxels += 1
                                    x_dist[round(abs(x2-x))] += 1
                                    y_dist[round(abs(y2-y))] += 1
                                    z_dist[round(abs(z2-z))] += 1
                                    xy_dist[round(sqrt(abs(x2-x) * abs(x2-x) + abs(y2-y) * abs(y2-y)))] += 1
                                    yz_dist[round(sqrt(abs(z2-z) * abs(z2-z) + abs(y2-y) * abs(y2-y)))] += 1
                                    xz_dist[round(sqrt(abs(x2-x) * abs(x2-x) + abs(z2-z) * abs(z2-z)))] += 1
                                    xyz_dist[round(sqrt(abs(x2-x) * abs(x2-x) + abs(y2-y) * abs(y2-y) + abs(z2-z) * abs(z2-z)))] += 1
            
            for i in range(101):
                x_dist[i] /= voxels
                y_dist[i] /= voxels
                z_dist[i] /= voxels
                xy_dist[i] /= voxels
                yz_dist[i] /= voxels
                xz_dist[i] /= voxels
                xyz_dist[i] /= voxels

            output[f'{key}, {key2}'] = {'x': x_dist, 'y': y_dist, 'z': z_dist, 'xy': xy_dist, 'yz': yz_dist, 'xz': xz_dist, 'xyz': xyz_dist}
    

    file = str(file).replace('ROI', 'hist')
    with open(file, 'w') as file:
        json.dump(output, file)