from json import dump
from os import listdir
from sys import argv

import numpy as np
from skimage import io, transform

from consts import *
from find import *


def main():
    # bamboo.py <data_path> <out_path> <show>
    data_path = argv[1]  # path to the data directory
    out_path = argv[2]   # path to the output directory
    show = len(argv) == 4 and argv[3] == "-s"  # plots the found bamboo lines
    filenames = [fn for fn in listdir(data_path) if fn.endswith(OBJ)]
    for fn in filenames:
        data_file_path = data_path + "/" + fn
        fragment = Render(data_file_path)
        bamboo_angle = find_bamboo_lines_angle(fragment, show)
        out_file_path = out_path + "/" + fn.removesuffix(OBJ) + JSON
        with open(out_file_path, mode='w') as out_file:
            dump(bamboo_angle, out_file)


def my_main():
    # ...09-...11 contains bamboo lines
    # ...12 doesn't contain bamboo lines
    path = "data/RPf_00311.obj"
    frag = Render(path, pose='backside')
    frag.plot()
    angle = find_bamboo_lines_angle(frag, show=True)
    if angle is not None:
        print(np.rad2deg(angle))
        frag.planar_rotation(angle)
        frag.plot()


def angles_poll(angles):
    real_angles = [a for a in angles if a is not None]
    if len(real_angles) > 0:
        return np.mean(real_angles)
    return None

    # Previous attempt:
    # real_angles = [a for a in angles if a is not None]
    # if len(real_angles) == 0:
    #     return None
    # angle_votes = np.zeros_like(real_angles)
    # for index_1, angle_1 in enumerate(real_angles):
    #     for index2, angle_2 in enumerate(real_angles):
    #         if index_1 != index2 and abs(angle_1 - angle_2) < epsilon:
    #             angle_votes[index_1] += 1
    # print(angle_votes)
    # return real_angles[np.argmax(angle_votes)]


def align_intact(frag_path, intact_path, show=False):
    frag = Render(frag_path, pose='backside')
    if show:
        frag.plot()
    angles = [find_bamboo_lines_angle(frag, show=show) for i in range(10)]
    angle = angles_poll(angles)
    if angle is not None:
        intact = io.imread(intact_path)
        rotated = transform.rotate(intact, np.rad2deg(angle))
        if show:
            io.imshow(aligned)
        return rotated
    return None


if __name__ == '__main__':
    for frag_id in ["09"]+([str(i) for i in range(10, 21)]):
        frag_path = "data/group39/RPf_003"+frag_id+".obj"
        intact_path = "data/group39/RPf_003"+frag_id+"_intact_mesh.png"
        aligned = align_intact(frag_path, intact_path)
        if aligned is not None:
            aligned = np.array(aligned * 255, dtype="uint8")
            io.imsave("out/RPf_003"+frag_id+"_aligned.png", aligned)
