from json import dump
from os import listdir
from sys import argv

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


if __name__ == '__main__':
    my_main()
