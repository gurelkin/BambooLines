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


def suspect_angles(ridge_map: np.array(list[bool]), plot=False) -> np.array:
    # try to find lines in the ridge map using Hough transform
    # angles = np.linspace(0, np.pi, 180, endpoint=False)
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


# def line_variance(angle, image, mask):
#     if angle < 0:
#         angle = 2 * np.pi + angle
#     height, width = image.shape
#     center_point = height // 2, width // 2
#     length = int(max(center_point) * np.sqrt(2))
#     p1 = center_point[1] + int(length * np.cos(angle)), center_point[0] + int(length * np.sin(angle))
#     p2 = center_point[1] - int(length * np.cos(angle)), center_point[0] - int(length * np.sin(angle))
#     rr, cc = skimage.draw.line(*p1, *p2)
#     rr, cc = np.clip(rr, 0, height - 1), np.clip(cc, 0, width - 1)
#     plot = np.zeros((height, width))
#     plot[rr, cc] = 1
#     plt.imshow(plot)
#     plt.show()
#     values = [image[index] for index in zip(rr, cc) if mask[index] == 1]
#     return np.var(values)
#
#
# def _line_masks(shape, angle, stride=64):
#     height, width = shape
#     center_i = height // 2
#     line = np.zeros(shape)
#     for j in range(width):
#         line[center_i, j] = 1
#     line = skimage.transform.rotate(line, np.rad2deg(angle))
#     lines = [line.copy()]
#     stride_x = stride / np.sin(angle)
#     stride_y = stride / np.cos(angle)
#     translation_matrix = skimage.transform.SimilarityTransform(translation=(stride_x, stride_y))
#     while np.any(line):
#         line = skimage.transform.warp(line, translation_matrix)
#         lines.append(line.copy())
#     translation_matrix = skimage.transform.SimilarityTransform(translation=(-stride_x, -stride_y))
#     line = lines[0]
#     while np.any(line):
#         line = skimage.transform.warp(line, translation_matrix)
#         lines.append(line.copy())
#     plot = np.zeros(shape)
#     for l in lines:
#         plot = np.where(l > 0, 1, plot)
#     plt.imshow(plot)
#     plt.show()


def perp(angle):
    return angle - np.sign(angle) * np.pi / 2


def line_masks(shape, angle, stride=64):
    height, width = shape
    lines = []
    for i in range(stride, height, stride):
        line = np.zeros(shape)
        line[i] = np.ones(width)
        lines.append(line)
    for index, line in enumerate(lines):
        line = skimage.transform.rotate(line, np.rad2deg(angle))
        lines[index] = np.where(line > 0, 1, 0)
    # all_lines_plot = np.zeros(shape)
    # for line in lines:
    #     all_lines_plot[line == 1] = 1
    # plt.imshow(all_lines_plot)
    # plt.show()
    return lines


def mask_variance(image, mask):
    values = image[mask == 1]
    if len(values) == 0:
        return None
    return np.var(values)


def find_bamboo_lines_angle(fragment: Render):
    # find the inner ridges (i.e., without the contour) in the fragment's depth map
    ridge_map = detect_ridges(fragment)
    # compute the angles of the bamboo lines
    angles = suspect_angles(ridge_map, plot=True)
    print(angles)
    # if there aren't any angles with sufficient score, we conclude there are no bamboo lines
    if len(angles) == 0:
        return None
    # choose the "most correct" angle
    depth_map = fragment.get_depth_map(normalize=True, invert=True)
    mask = fragment.get_mask()
    parallel_lines = [line_masks(depth_map.shape, perp(-a)) for a in angles]
    perp_lines = [line_masks(depth_map.shape, -a) for a in angles]
    parallel_vars = [np.mean([mask_variance(depth_map, mask*line)
                              for line in lines
                              if np.max(mask*line) > 0])
                     for lines in parallel_lines]
    perp_vars = [np.mean([mask_variance(depth_map, mask * line)
                          for line in lines
                          if np.max(mask*line) > 0])
                 for lines in perp_lines]
    print(parallel_vars)
    print(perp_vars)
    scores = np.abs(np.subtract(perp_vars, parallel_vars))
    print(scores)
    return perp(-angles[np.argmax(scores)])

    # parallel_variances = [line_variance(a, depth_map, mask) for a in angles]
    # perpendicular_variances = [line_variance(perp(a), depth_map, mask) for a in angles]

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