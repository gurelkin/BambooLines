from render import *
from consts import *


def detect_ridges(depth_map: np.ndarray, mask: np.ndarray, show=False) -> np.array:
    """
    Finds the inner ridges (as in without the contour) of a fragment's (normalized and inverted) `edge_map`.

    :param depth_map: depth map od the fragment's backside.
    :param mask: a mask to filter ot the background.
    :param show: if True, plots the output ridge map.
    :return: an image of the inner ridges of the fragment.
    """
    # filter the ridges in the depth map
    hess = 1 - skimage.filters.hessian(depth_map)
    # remove the outermost contour from the ridge map
    contours = skimage.measure.find_contours(hess)
    outermost_contour = max(contours, key=len)
    rr, cc = np.round(outermost_contour[:, 0]), np.round(outermost_contour[:, 1])
    inner_mask = np.zeros_like(hess)
    inner_mask[rr.astype(int), cc.astype(int)] = 1
    inner_mask = skimage.morphology.dilation(inner_mask, skimage.morphology.disk(8))
    inner_mask = (1 - inner_mask) * mask
    ridge_map = np.where(inner_mask == 1, hess, 0)
    # for visual inspection
    if show:
        plt.imshow(ridge_map, cmap='gray')
        plt.show()
    return ridge_map


def suspect_angles(ridge_map, show=False) -> np.ndarray:
    """
    Selects candidate angles for bamboo lines from a `ridge_map`.

    :param ridge_map: output of detect_ridges function.
    :param show: plots the lines found in the Hough transform.
    :return: a list of angles that might be the angles of the bamboo lines.
    """
    # try to find lines in the ridge map using Hough transform
    hough_space, thetas, rhos = skimage.transform.hough_line(ridge_map)
    peaks = skimage.transform.hough_line_peaks(hough_space, thetas, rhos, threshold=HOUGH_SCORE_THRESHOLD)
    # for visual inspection
    if show:
        plt.imshow(ridge_map, cmap='gray')
        for score, angle, dist in zip(*peaks):
            (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
            plt.axline((x0, y0), slope=np.tan(angle + np.pi / 2))
        plt.show()
    # return the angles of the hough-space peak lines (that have sufficient score)
    return peaks[1]


def perp(angle):
    """
    Calculates the perpendicular of an `angel` in the range [-pi/2, pi/2].

    :param angle: input angle.
    :return: the perpendicular angle, still within range.
    """
    return angle - np.sign(angle) * np.pi / 2


# ========== older version of line_masks function ==========
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
# ==========================================================


def line_masks(shape: tuple[int, int], angle: float, stride=64):
    """
    Creates (`shape[0]` // `stride`) masks, each contains one line tilted in the given `angle`. The spacing between the
    lines is determined by `stride`.

    :param shape: (height, width) of the mask.
    :param angle: the angle in which to tilt the lines (between -pi/2 and pi/2).
    :param stride: space between two lines (in pixels).
    :return: list of line masks.
    """
    height, width = shape
    lines = []
    # create horizontal lines
    for i in range(stride, height, stride):
        line = np.zeros(shape)
        line[i] = np.ones(width)
        lines.append(line)
    # tilt the lines and make sure the mask remains binary
    for index, line in enumerate(lines):
        line = skimage.transform.rotate(line, np.rad2deg(angle))
        lines[index] = np.where(line > 0, 1, 0)
    # ========== uncomment to plot the calculated lines ==========
    # all_lines_plot = np.zeros(shape)
    # for line in lines:
    #     all_lines_plot[line == 1] = 1
    # plt.imshow(all_lines_plot)
    # plt.show()
    # ============================================================
    return lines


def mask_variance(image: np.ndarray, mask: np.ndarray):
    """
    Calculates the variance of an `image`'s pixels defined by `mask`.

    :param image: grayscale image from which to calculate the variance.
    :param mask: binary ndarray in the shape of the image.
    :return: variance of image[mask == 1] or None if mask is all zeros.
    """
    values = image[mask == 1]
    if len(values) == 0:
        return None
    return np.var(values)


def find_bamboo_lines_angle(fragment: Render, show=False):
    """
    Detects the angle of the bamboo lines in the given `fragment`, if exist.

    :param fragment: input fragment.
    :param show: if True, plots the results of some sub-procedures.
    :return: the bamboo lines angle or None if no bamboo lines were found; that is, the (acute) angle the bamboo lines
    create with the positive direction of the x-axis.
    """
    # find the inner ridges (i.e., without the contour) in the fragment's depth map
    depth_map = fragment.get_depth_map(normalize=True, invert=True)
    mask = fragment.get_mask()
    ridge_map = detect_ridges(depth_map, mask, show=show)
    # compute the angles of the bamboo lines
    angles = suspect_angles(ridge_map, show=show)
    # if there aren't any angles with sufficient score, we conclude there are no bamboo lines
    if len(angles) == 0:
        return None
    elif len(angles) == 1:
        return perp(-angles[0])
    # choose the "most correct" angle; that is, the angle that have the highest difference between its parallel
    # line masks variance and its perpendicular line masks variance
    parallel_lines = [line_masks(depth_map.shape, perp(-a)) for a in angles]
    perp_lines = [line_masks(depth_map.shape, -a) for a in angles]
    parallel_vars = [np.mean([mask_variance(depth_map, mask*line)
                              for line in lines
                              if np.max(mask*line) > 0])
                     for lines in parallel_lines]
    perp_vars = [np.mean([mask_variance(depth_map, mask*line)
                          for line in lines
                          if np.max(mask*line) > 0])
                 for lines in perp_lines]
    scores = np.abs(np.subtract(perp_vars, parallel_vars))
    return perp(-angles[np.argmax(scores)])
