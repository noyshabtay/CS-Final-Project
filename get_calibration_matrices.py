import os
from utils import printProgressBar
import numpy as np
import cv2 as cv
from pathlib import Path
import re


# There are 2 main functions:
# 1. cameras_frames_extractor(path, jumping_density = 20) - for extracting frames from a video calibration video.
# after extraction you'll need to keep only the frames with full and clear chessboard.
# 2. create_new_calibration_matrices_file(path) - for creating a ptyhon matrices file from the folders containing the chess borad frames.

def init_video(video_path):
    """
    Given the path of a video, prepares the flux and checks that everything works as attended.
    """
    capture = cv.VideoCapture(video_path)
    if not capture.isOpened():
        print('Failed opening capture')
        return None

    fps = capture.get(cv.CAP_PROP_FPS)
    if fps != 0:
        return capture
    else:
        return None


def from_video_to_frames(parent_dir, file, jumping_density=20):
    """
    Arguments: 
    parent_dir- path to the folder of the selected video.
    file- video file name.
    jumping_density- int, determines how often we sample frames from the video.
    Effects:
    crates a new folder named as the video file in parent_dir with that video's frames.
    Returns:
    None
    """
    video_name = file.replace('.mp4', '')
    dir_path = os.path.join(parent_dir, video_name)
    file_path = os.path.join(parent_dir, file)
    video = init_video(file_path)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    nFrames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    print(f"reading video {video_name} ")
    frame_to_capture = 0
    for i in range(nFrames):
        _, image = video.read()
        if i == frame_to_capture:
            cv.imwrite(os.path.join(dir_path, f"frame{i}_{video_name}.jpg"), image)  # save frame as JPEG file
            frame_to_capture += jumping_density
        printProgressBar(i, nFrames + 1, prefix='Progress: ')


def cameras_frames_extractor(path, jumping_density=20):
    for file in os.listdir(path):
        if file.endswith(".mp4"):
            from_video_to_frames(path, file, jumping_density)
    print()
    print("finished extracting frames from all videos.")
    print("now, in each folder, manully delete frames *not* containing the chess board fully or clearly!")
    print("it's best that each folder will contain around 150-200 frames.")


class CalibrationException(Exception):
    pass


def get_distortion_matrix(chkr_im_path: Path, rows=6, cols=9):
    """
    Finds the undistortion matrix of the lens based on multiple images
    with checkerboard. It's possible to implement this function using Aruco markers as well.
    :param: chkr_im_path - path to folder with images with checkerboards
    :param: rows - number of rows in checkerboard
    :param: cols - number of cols in checkerboard
    :return: numpy array: camera matrix, numpy array: distortion coefficients
    """

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    image_paths = list(chkr_im_path.iterdir())

    # drawings = []
    # imgs = []
    print("1/3: loading frames")
    for fname in image_paths:
        img = cv.imread(str(fname))
        shape = img.shape
        # imgs.append(img)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (cols, rows), None)

        # If found, add object points, image points (after refining them)
        if ret is True:
            objpoints.append(objp)

            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            # img = cv.drawChessboardCorners(img, (cols,rows), corners2,ret)
            # drawings.append(img)
    print("2/3: calculating calibration matrices, please wait...")
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, shape[::-1][1:], None, None
    )

    if not ret:
        raise CalibrationException("Error finding distortion matrix")
    print("3/3: done.")
    return mtx, dist


def matrix_formatter(mtx):
    """
    helper funtion for prinitng a matrix.
    """
    return re.sub("\s+", ", ", str(mtx).strip())


def create_new_calibration_matrices_file(path, new_matrices_file_name):
    """
    Arguments: 
    path- path to the parent folder containing folders of frames for each camera.
    !!! Note that these folders should contain only relevant frames (incl. chess board fully and clearly). !!!
    Effects:
    crates a new python file called matrices including all the calibration matrices.
    Returns:
    None
    """
    camera_types = ["TOP", "BACK", "LEFT", "RIGHT"]
    with open(f'{new_matrices_file_name}.py', 'w') as py_file:
        print("import numpy as np", file=py_file)
        print("from camera import Camera", file=py_file)
        print("", file=py_file)
        for type in camera_types:
            for file in os.listdir(path):
                file_path = os.path.join(path, file)
                if type.lower() in file and os.path.isdir(file_path):
                    print(f"*** calculating calibration matrices for {type} camera ***")
                    mtx, dist = get_distortion_matrix(Path(file_path))
                    print(f"MTX_{type} = np.array(", file=py_file)
                    print("\t" + matrix_formatter(mtx), file=py_file)
                    print(")", file=py_file)
                    print("", file=py_file)
                    print(f"DIST_{type} = np.array(", file=py_file)
                    print("\t" + matrix_formatter(dist), file=py_file)
                    print(")", file=py_file)
                    print("", file=py_file)
        print("matrices = {", file=py_file)
        for type in camera_types:
            print(f"\tCamera.{type}: (MTX_{type}, DIST_{type}),", file=py_file)
        print("}", file=py_file)


if __name__ == '__main__':
    path = "C:/Users/noy_s/Documents/gaze_vector/pose_analysis/data/calib/undistortion_12_07"
    cameras_frames_extractor(path, jumping_density=1000)
    create_new_calibration_matrices_file(path)
