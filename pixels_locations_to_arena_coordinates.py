import argparse
import cv2
from utils import printProgressBar
import os
import csv
from os import path

from coordinate_calculator import CoordinateCalculator
from arena import load_arena, file_to_camera
from utils import init_video


def init_videos(exp_name, videos_dir, video_name, out_dir, camera):
    print('visualizing analysis with video:', video_name)
    video = init_video(path.join(videos_dir, video_name))
    out_file = "{0}/{1}_{2}_detected.mp4".format(out_dir, exp_name, camera.value)
    print('writing analysis video:', out_file)

    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)
    output = cv2.VideoWriter(out_file, -1, fps, size)
    return video, output


def process_camera_input(camera, video, out, points_by_camera, cc, likelihood, show):
    skipped = 0
    print('analyzing camera:', camera)
    camera_points = points_by_camera[camera]

    n_points = len(camera_points)
    for i in range(n_points):
        printProgressBar(i + 1, n_points, prefix='Progress: ')
        points = points_by_camera[camera][i][0]
        p_likelihood = points_by_camera[camera][i][1]

        frame = None
        if video is not None:
            result, frame = video.read()
            font = cv2.FONT_HERSHEY_SIMPLEX
            height, width, _ = frame.shape
            cv2.putText(frame, 'likelihood {:.3f}'.format(p_likelihood), (20, 50), font, 0.5, (0, 255, 0), 1,
                        cv2.LINE_AA)
            cv2.putText(frame, f'# frame {i}', (20, 30), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        if float(p_likelihood) > likelihood:
            cc.calculate_camera_coordinates(camera, points, frame)
        else:
            cc.add_skipped_frame(camera)
            skipped += 1

        if frame is not None:
            if show:
                cv2.imshow('frame', frame)
                cv2.waitKey(1)
            out.write(frame)
    if out is not None:
        out.release()

    cv2.destroyAllWindows()


def process_video_with_given_points(exp_name, points_by_camera, out_dir, videos_dir, show, likelihood=0.9, arena=None):
    cc = CoordinateCalculator(arena=arena)
    videos = []
    if videos_dir is not None:
        videos = os.listdir(videos_dir)
    for camera in points_by_camera:
        video = None
        out = None
        for v in videos:
            if v.find(camera.value) != -1 and v.find(exp_name) != -1:
                video, out = init_videos(exp_name, videos_dir, v, out_dir, camera)
                break

        process_camera_input(camera, video, out, points_by_camera, cc, likelihood, show)
    cc.save_3d_points(exp_name, out_dir) #os.path.join(out_dir, 'points')


def read_points_csv(csv_path, timestamp, filter):
    experiment_csvs = [file for file in os.listdir(csv_path) if file.find(timestamp) != -1
                       and (filter is None or filter in file)]
    print('read experiments csvs:', experiment_csvs)
    points_by_camera = {}
    for file in experiment_csvs:
        camera = file_to_camera(file)
        points_by_camera[camera] = []
        with open(path.join(csv_path, file), 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                if not row[0].split('.')[0].isdigit():
                    continue
                frame_points = []
                for i in range(4):
                    frame_points.append((int(float(row[1 + (i * 3)])), int(float(row[2 + (i * 3)]))))
                likelihoods = min([float(row[3 + (i * 3)]) for i in range(4)])
                points_by_camera[camera].append((frame_points, likelihoods))

    return points_by_camera


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, default=os.path.join("points"),
                        help="directory of the output ")
    parser.add_argument("--csv", type=str)
    parser.add_argument("-m", "--matrices_dir", type=str, default="matrices",
                        help="directory of the input matrices")
    parser.add_argument("-t", "--timestamp", type=str,
                        help="the timestamp of the experiment")
    parser.add_argument("-l", "--likelihood", type=float, default=0.9,
                        help="the minimum likelihood of points to use")
    parser.add_argument("-f", "--filter", type=str,
                        help="file name filter")
    parser.add_argument("-v", "--videos_dir", type=str,
                        help="the videos dir if you want to visualize result")
    parser.add_argument("-s", "--show_video", type=bool, default=False)


    parser.set_defaults(sample=False)

    args = parser.parse_args()
    cameras_points = read_points_csv(args.csv, args.timestamp, args.filter)
    arena = load_arena(args.matrices_dir, "")  # args.timestamp)
    process_video_with_given_points(args.timestamp, cameras_points, args.output, args.videos_dir, args.show_video, arena=arena, likelihood=float(args.likelihood))


if __name__ == '__main__':
    main()
