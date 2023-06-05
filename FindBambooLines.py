from os import listdir
from sys import argv
from json import dump

from FragmentDepthRender import *
from globals import *


def find_bamboo_lines(render: FragmentDepthRender, show=False) -> list[tuple[float, float]]:
    return [(0, 0)]


def main():
    # bamboo <data_path> <out_path> <show>
    data_path = argv[1]  # path to the data directory
    out_path = argv[2]   # path to the output directory
    show = len(argv) == 4 and argv[3] == "-s"  # plots the found bamboo lines
    filenames = [filename for filename in listdir(data_path) if filename.endswith(OBJ)]
    for fn in filenames:
        data_file_path = data_path + "/" + fn
        fragment = FragmentDepthRender(data_file_path)
        bamboo_lines = find_bamboo_lines(fragment, show)
        out_file_path = out_path + "/" + fn.removesuffix(OBJ) + JSON
        with open(out_file_path, 'w') as out_file:
            dump(bamboo_lines, out_file)


if __name__ == '__main__':
    main()
