import numpy as np
import scipy
import skimage
from matplotlib import pyplot as plt
import trimesh
import open3d as o3d
from tqdm import tqdm
from render import *
from consts import *


def detect_ridges(fragment: Render) -> np.array:
    # filter the ridges in the fragment's depth map
    depth_map = fragment.get_depth_map(normalize=True, invert=True)
    hess = 1 - skimage.filters.hessian(depth_map)
    # remove the outermost contour from the ridge map
    contours = skimage.measure.find_contours(hess)
    outermost_contour = max(contours, key=len)
    rr, cc = np.round(outermost_contour[:, 0]), np.round(outermost_contour[:, 1])
    mask = np.zeros_like(hess)
    mask[rr.astype(int), cc.astype(int)] = 1
    mask = skimage.morphology.dilation(mask, skimage.morphology.disk(8))
    mask = (1 - mask) * fragment.get_mask()
    return np.where(mask == 1, hess, 0)


def suspect_angles(ridge_map: np.array(list[bool]), plot=False) -> np.array(list[float]):
    # try to find lines in the ridge map using Hough transform
    hough_space, thetas, rhos = skimage.transform.hough_line(ridge_map)
    peaks = skimage.transform.hough_line_peaks(hough_space, thetas, rhos, threshold=HOUGH_SCORE_THRESHOLD)
    # for visual inspection
    if plot:
        plt.imshow(ridge_map, cmap='gray')
        for score, angle, dist in zip(*peaks):
            (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
            plt.axline((x0, y0), slope=np.tan(angle + np.pi / 2))
        plt.show()
    # return the angles of the hough-space peak lines (that have sufficient score)
    return peaks[1]


def make_line_masks(angle, shape, num_lines=16, spacing=64):
    cos_angle, sin_angle = np.cos(angle), np.sin(angle)
    lines = []

    center_x = (shape[1] - 1) / 2
    center_y = (shape[0] - 1) / 2
    start_x = center_x - (num_lines // 2) * spacing * np.sin(angle)
    start_y = center_y + (num_lines // 2) * spacing * np.cos(angle)

    for i in range(num_lines):
        line_x = np.arange(shape[1]) * cos_angle - (i * spacing) * sin_angle
        line_y = np.arange(shape[0]) * sin_angle + (i * spacing) * cos_angle
        line_x_indices = np.round(line_x).astype(int)
        line_y_indices = np.round(line_y).astype(int)

        if len(line_x_indices) != len(line_y_indices):
            min_len = min(len(line_x_indices), len(line_y_indices))
            line_x_indices = line_x_indices[:min_len]
            line_y_indices = line_y_indices[:min_len]

        valid_indices = np.logical_and.reduce((
            line_x_indices >= 0,
            line_x_indices < shape[1],
            line_y_indices >= 0,
            line_y_indices < shape[0]
        ))
        line_x_indices = line_x_indices[valid_indices]
        line_y_indices = line_y_indices[valid_indices]
        lines.append([line_y_indices, line_x_indices])

    return lines


def find_bamboo_lines_angle(fragment: Render):
    # find the inner ridges (i.e., without the contour) in the fragment's depth map
    ridge_map = detect_ridges(fragment)
    # compute the angles of the bamboo lines
    angles = suspect_angles(ridge_map, plot=True)
    print(angles * (180 / np.pi))
    # if there aren't any angles with sufficient score, we conclude there are no bamboo lines
    if len(angles) == 0:
        return None
    # if there's only one candidate angle, return that one
    if len(angles) == 1:
        return angles[0]
    # choose the "most correct" angle
    scores = np.zeros_like(angles)
    depth_map = fragment.get_depth_map(normalize=True)
    line_masks = [make_line_masks(a, depth_map.shape) for a in angles]
    lines = np.zeros_like(depth_map)
    for lm in line_masks:
        for [rr, cc] in lm:
            lines[rr, cc] = 1
    plt.imshow(lines)
    plt.show()
    # plt.imshow(hess)
    # plt.show()
    # hough_suspects(hess, plot=True)
    # canny = skimage.feature.canny(depth_map)
    # canny = np.where(mask == 0, 0, canny)
    # plt.imshow(canny)
    # plt.show()
    # hough_suspects(canny, plot=True)
    # ridge_mask = np.where(depth_map > 0.9, 1, 0)
    # plt.imshow(ridge_mask)
    # plt.show()
    # baz = np.where(mask == 1, hess, 1)
    # plt.imshow(baz)
    # plt.show()