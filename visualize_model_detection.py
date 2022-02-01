import argparse
import cv2 as cv
from utils import printProgressBar
from skspatial.objects import Point, Vector
import os
import csv
import numpy as np
from os import path

from matrices import matrices
from arena import load_arena, file_to_camera
from utils import init_video
from video_drawer import draw_direction

import pandas as pd


def init_videos(exp_name, videos_dir, video_name, out_dir, camera):
    print('visualizing analysis with video:', video_name)
    video = init_video(path.join(videos_dir, video_name))
    out_file = "{0}/{1}_{2}_visualize.mp4".format(out_dir, exp_name, camera.value)
    print('writing analysis video:', out_file)

    fps = video.get(cv.CAP_PROP_FPS)
    width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)
    output = cv.VideoWriter(out_file, -1, fps, size)
    return video, output


def draw_given_points(points, out_dir, videos_dir, exp_name, arena):
    videos = [v for v in os.listdir(videos_dir) if v.find(exp_name) >= 0]
    x_points = points.filter(regex='CenterX')
    x_points = x_points[x_points.columns[0]]
    y_points = points.filter(regex='CenterY')
    y_points = y_points[y_points.columns[0]]
    z_points = points.filter(regex='CenterZ')
    z_points = z_points[z_points.columns[0]]
    dx_points = points.filter(regex='Direction').filter(regex='X')
    dx_points = dx_points[dx_points.columns[0]]
    dy_points = points.filter(regex='Direction').filter(regex='Y')
    dy_points = dy_points[dy_points.columns[0]]
    dz_points = points.filter(regex='Direction').filter(regex='Z')
    dz_points = dz_points[dz_points.columns[0]]

    for v in videos:
        print('visualize points on video: ', v)
        camera = file_to_camera(v)
        video, out = init_videos(exp_name, videos_dir, v, out_dir, camera)
        r, t = arena.get_camera_matrices(camera)
        matrix, dist = matrices[camera]
        n_points = len(points)
        for i in range(n_points):
            printProgressBar(i + 1, n_points, prefix='Progress: ')
            result, frame = video.read()
            if np.isnan(x_points[i]) or np.isnan(y_points[i]) or np.isnan(z_points[i]):
                continue
            point = (x_points[i] ,y_points[i], z_points[i])
            direction = (dx_points[i] ,dy_points[i], dz_points[i])
            dpoint = Point(point) + Vector(direction)
            projected, _ = cv.projectPoints(np.array([point, dpoint]), r, t, matrix, dist)
            draw_direction(projected, frame)
            cv.imshow('frame', frame)
            cv.waitKey(1)
            out.write(frame)
    out.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, default=os.path.join("results"),
                        help="directory of the output ")
    parser.add_argument("--csv", type=str)
    parser.add_argument("-m", "--matrices_dir", type=str, default="matrices",
                        help="directory of the input matrices")
    parser.add_argument("-v", "--videos_dir", type=str,
                        help="the video path")
    parser.add_argument("-t", "--timestamp", type=str,
                        help="the timestamp of the experiment")

    args = parser.parse_args()
    camera_points = pd.read_csv(args.csv)
    arena = load_arena(args.matrices_dir, "")#args.timestamp)
    draw_given_points(camera_points, args.output, args.videos_dir, args.timestamp, arena=arena)


if __name__ == '__main__':
    main()
