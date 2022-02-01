import argparse
import cv2
import pickle
import os
from os import path
import matplotlib.pyplot as plt

import numpy as np
from arena import load_arena, file_to_camera
from camera import Camera
from utils import init_video
from coordinate_calculator import CoordinateCalculator

# Arena configuration values.
arena_width = 68.3
arena_length = 97.0
arena_height = 45
board_arena_x_gap = 6.35
board_corner = {
    'A': (0, 0),
    'B': (arena_length - board_arena_x_gap, arena_width),
}
chess_board_z = 1
board_x_inner_corners = 9
board_y_inner_corners = 6
square_size = 2.5


def plot_points(img, points, width, height, color):
    for p in points:
        p = p.reshape(-1).tolist()
        if p[0] <= 0 or p[1] <= 0 or p[0] >= width or p[1] >= height:
            continue
        p = (int(p[0]), int(p[1]))
        cv2.circle(img, p, 2, color)


def plot_arena_axis(frame, world_to_pixels):
    arena_width_lines = list(range(0, int(arena_width + 1), 10))
    arena_width_lines.append(arena_width)
    arena_length_lines = list(range(0, int(arena_length + 1), 10))
    arena_length_lines.append(arena_length)
    x_points = [(i, j, 1) for i in range(int(arena_length+1)) for j in arena_width_lines]
    y_points = [(j, i, 1) for i in range(int(arena_width + 1)) for j in arena_length_lines]
    z_points = [(x, y, i) for i in range(1, arena_height + 1) for x in [0, arena_length] for y in [0, arena_width]]

    x_pixels = [world_to_pixels(x) for x in x_points]
    y_pixels = [world_to_pixels(y) for y in y_points]
    z_pixels = [world_to_pixels(z) for z in z_points]
    height, width, _ = frame.shape
    plot_points(frame, x_pixels, width, height, (0, 0, 255))
    plot_points(frame, y_pixels, width, height, (0, 255, 0))
    plot_points(frame, z_pixels, width, height, (255, 0, 0))


def get_board_visible_points(board, square_size):
    board_visible = input(f'enter is board {board} visible (yes/no): ') == "yes"
    arena_points = []
    point_names = []

    if not board_visible:
        return [], []

    possible_inner_corners = [
            (1, 1), (1, board_y_inner_corners),
    ]
    for i in range(2, board_y_inner_corners + 1):
        possible_inner_corners.append((i, i))
    possible_inner_corners.extend([
        (board_x_inner_corners, 1),
        (board_x_inner_corners, board_y_inner_corners)
    ])

    all_visible = input(f'is inner corners {possible_inner_corners} in '
        f'board {board}, are all visible? (yes/no)') == 'yes'
    for p in possible_inner_corners:
        i, j = p
        if not all_visible:
            visible = input(f'is inner corner ({i},{j}) in board {board}, visible? (yes/no)') == 'yes'
            if not visible:
                continue
        c = get_point_coordinates(1, 1, square_size,
                                      i, j, board)
        arena_points.append(c)
        point_names.append((board, i, j))
    return arena_points, point_names


def get_camera_matrix_manual(camera, frame, videos_dir, manual_points_file):
    plt.figure(figsize=(12, 8))
    plt.title(f'camera {camera.value}')
    plt.imshow(frame)
    plt.show()
    arena_points = []
    point_names = []

    for board in board_corner.keys():
        b_arena_points, b_point_names = get_board_visible_points(board, square_size)
        arena_points.extend(b_arena_points)
        point_names.extend(b_point_names)

    points = select_points(point_names, camera, frame)
    return calculate_matrix(camera, arena_points, points, frame, videos_dir)


def select_points(point_names, camera, frame):
    cv2.imshow(camera.value, frame)
    points = []

    def onclick(event, x, y, flags, params):
        if event != cv2.EVENT_RBUTTONDOWN:
            return

        points.append([x, y])
        if len(points) == len(point_names):
            cv2.destroyAllWindows()
            return
        print('right click point {}'.format(point_names[len(points)]))

    cv2.setMouseCallback(camera.value, onclick)
    print('right click point {}'.format(point_names[len(points)]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return points


def get_point_coordinates(point_zero_line, point_zero_column, square_size,
                          i, j, board):
    board_corner_x, board_corner_y = board_corner[board]
    squares_in_x = i if point_zero_line == 1 else point_zero_line - (i-1)
    squares_in_y = j if point_zero_column == 1 else point_zero_column - (j-1)
    board_direction = 1 if board == 'A' else -1

    x = board_corner_x + (squares_in_x * square_size * board_direction)
    y = board_corner_y + (squares_in_y * square_size * board_direction)
    z = chess_board_z
    return x, y, z


def calculate_matrix(camera, arena_points, points, frame, videos_dir):
    display_frame = frame.copy()
    cc = CoordinateCalculator(model_points=np.array(arena_points))
    world_to_pixels, r, t = cc.get_world_to_pixels_translation(camera, points)
    plot_arena_axis(display_frame, world_to_pixels)
    plt.figure(figsize=(12, 8))
    plt.imshow(display_frame)
    plt.show()

    arena_axis_file_name = f'arena_axis_{camera}.png'
    arena_axis_file_path = "{0}/{1}".format(videos_dir, arena_axis_file_name)
    cv2.imwrite(arena_axis_file_path, display_frame)
    return r, t


def get_camera_matrix(videos_dir, video_name, manual_points_file):
    camera = file_to_camera(video_name)
    video_file = "{0}/{1}".format(videos_dir, video_name)
    video = init_video(video_file)

    result, frame = video.read()
    if not result:
        print('problem reading from video: frame {0}, result {1}'.format(i, result))
        return

    display_frame = frame.copy()
    # try to detect chessboard in frame.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    img = frame.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (board_x_inner_corners, board_y_inner_corners), None)
    chess_corners = []
    chess_3d_corners = []
    if not ret:
        print('Chessboard was not detected, Need to do manual calibration')
        should_continue = input('Enter yes for manual calibration: ')
        if should_continue != "yes":
            print(f'Skipping f{camera} matrix calculation')
            return
        return get_camera_matrix_manual(camera, frame, videos_dir, manual_points_file)

    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (board_x_inner_corners, board_y_inner_corners), corners2, ret)
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.title(f'image with detected chess board, camera {camera.value}')
        plt.show()

        for i in range(len(corners2)):
            c = corners2[i]
            corner = c.reshape(-1)
            chess_corners.append(tuple(corner))
            frame = cv2.circle(frame, tuple(corner.astype(int)), 1, (0, 0, 255), 2)
            cv2.putText(frame, f'{i}', tuple(corner.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

        plt.figure(figsize=(12, 8))
        plt.imshow(frame)
        plt.show()

        print('initiating calibration based on chess board coordinates:')
        board = input('enter detected board (A for (0,0,0), B otherwise): ')

        point_zero_line = int(input(f'enter the line (x-axis) of point 0 in the photo (1-{board_x_inner_corners}:'))
        point_zero_column = int(input(f'enter the column (y-axis) of point 0 in the photo (1-{board_y_inner_corners}:'))
        square_size = 2.5

        # corners are ordered row by row, left to right in every row. there are ten rows.
        for j in range(1, board_y_inner_corners + 1):
            for i in range(1, board_x_inner_corners + 1):
                c = get_point_coordinates(point_zero_line, point_zero_column, square_size,
                                          i, j, board)
                chess_3d_corners.append(c)

        second_board = 'A' if board == 'B' else 'B'
        arena_points, point_names = get_board_visible_points(second_board, square_size)
        chess_3d_corners.extend(arena_points)
        points = select_points(point_names, camera, frame)
        chess_corners.extend(points)

        cc = CoordinateCalculator(model_points=np.array(chess_3d_corners))
        world_to_pixels, r, t = cc.get_world_to_pixels_translation(camera, chess_corners)
        plot_arena_axis(display_frame, world_to_pixels)
        plt.figure(figsize=(12, 8))
        plt.imshow(display_frame)
        plt.show()
        arena_axis_file_name = f'arena_axis_{camera}.png'
        arena_axis_file_path = "{0}/{1}".format(videos_dir, arena_axis_file_name)
        cv2.imwrite(arena_axis_file_path, display_frame)

        return r, t


def get_extrinsic_matrix(r, t):
    extrinsic_matrix = np.zeros((3, 4), dtype=float)
    extrinsic_matrix[:3, :3] = r
    extrinsic_matrix[:, 3] = t.reshape((3,))
    return extrinsic_matrix


def camera_to_world(r, t, point):
    point = point - t
    m = np.linalg.inv(np.matrix(r))
    return m * point


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--videos_dir", type=str, default=os.path.join("videos", "undistorted"),
                        help="directory of the input videos")
    parser.add_argument("-m", "--matrices_dir", type=str, default="matrices",
                        help="directory of output matrices")
    parser.add_argument("-p", "--manual_points", type=str,
                        help="path of manual chosen points data")
    parser.add_argument("-t", "--timestamp", type=str,
                        help="the timestamp of the experiment")

    args = parser.parse_args()

    experiment_videos = [file for file in os.listdir(args.videos_dir) if file.find(args.timestamp) != -1]
    for video in experiment_videos:
        print(f'analyzing {video}')
        camera_matrix = get_camera_matrix(args.videos_dir, video, args.manual_points)
        print(f'done analyzing {video}')
        matrix_file_path = path.join(args.matrices_dir, f'{video.split(".")[0]}_matrix.data')
        with open(matrix_file_path, 'wb') as f:
            pickle.dump(camera_matrix, f)
            print(f'writing camera matrix to: {matrix_file_path}')

    print('Sanity check: printing location of cameras relatively to the arena')
    arena = load_arena(args.matrices_dir, args.timestamp)
    camera_point = np.array([0, 0, 0]).reshape((3, 1))
    for c in Camera:
        p = arena.translate_point_to_world(c, camera_point)
        print(c, p)


if __name__ == '__main__':
    main()
