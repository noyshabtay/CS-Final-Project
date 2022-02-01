import numpy as np
import cv2 as cv
from skspatial.objects import Plane
from skspatial.objects import Points, Point, Vector
from os import path
from datetime import datetime
import csv
from camera import Camera
from video_drawer import draw_direction, draw_model
from matrices import matrices
from arena import in_arena


MODEL_POINTS = np.array(
    [
        (-1.151, .9831, -.3044),  # FL
        (.9508, 1.1708, -.3701),  # FR
        (-1.4692, -.3132, -.4578),  # BL
        (1.4218, -.1685, -.4578)  # BR
    ]
)

MODEL_CENTER = (0, 0, 0)


def get_head_direction_point(model_points, model_center):
    '''
    :param model_points: 3d coordinates of the 4 led centers.
    :return: the direction of the vector that connects the projection of the
             model center on the leds plane and the point between the two front leds.
             This might be changed in the future, depends on the calibration of the
             head and model direction.
    '''
    points = Points(model_points)
    plane = Plane.best_fit(points)
    center_point_projected = plane.project_point(Point(model_center))
    center_front_point = Points([MODEL_POINTS[0], MODEL_POINTS[1]]).centroid()
    vector = Vector.from_points(center_point_projected, center_front_point)
    head_direction = vector.unit()
    return head_direction, center_point_projected


HEAD_DIRECTION_POINT, PROJECTED_MODEL_CENTER = get_head_direction_point(MODEL_POINTS, MODEL_CENTER)


class CoordinateCalculator:
    def __init__(self, model_points=MODEL_POINTS, arena=None):
        self.arena = arena
        self.model_points = model_points

        self.points = {}
        self.centers = {}
        self.directions = {}
        for c in Camera:
            self.points[c] = []
            self.centers[c] = []
            self.directions[c] = []

    def add_skipped_frame(self, camera):
        self.points[camera].append([])
        self.centers[camera].append([])
        self.directions[camera].append([])

    def calculate_world_coordinates(self, camera, extrinsic_matrix, p):
        homogeneous_p = np.zeros((4, 1), dtype=float)
        homogeneous_p[:3] = p.reshape(3, 1)
        homogeneous_p[3] = 1

        camera_position = np.matmul(extrinsic_matrix, homogeneous_p)
        if self.arena is not None:
            camera_position = self.arena.translate_point_to_world(camera, camera_position)
        return camera_position

    def calculate_camera_coordinates(self, camera, centers, img = None):
        if img is not None:
            draw_model(centers, img)

        image_points = np.array(centers, dtype="double")
        matrix, dist = matrices[camera]
        (success, rotation_vector, translation_vector) = cv.solvePnP(self.model_points, image_points,
                                                                     matrix, dist)
        if not success:
            print('failed solving pnp')
            self.add_skipped_frame(camera)
            return

        # translate world coordinates to camera coordinates.
        camera_positions = []
        i = 0
        rotation_matrix, _ = cv.Rodrigues(rotation_vector)

        extrinsic_matrix = np.zeros((3, 4), dtype=float)
        extrinsic_matrix[:3, :3] = rotation_matrix
        extrinsic_matrix[:, 3] = translation_vector.reshape((3,))

        for pos in self.model_points:
            homogeneous_p = np.zeros((4, 1), dtype=float)
            homogeneous_p[:3] = pos.reshape(3, 1)
            homogeneous_p[3] = 1

            camera_position = np.matmul(extrinsic_matrix, homogeneous_p)
            if self.arena is not None:
                camera_position = self.arena.translate_point_to_world(camera, camera_position)
            camera_positions.append(np.asarray(camera_position).reshape(-1))
            i += 1

        model_center_homogeneous_world_coordinates = np.array([0, 0, 0, 1])
        model_center = np.matmul(extrinsic_matrix, model_center_homogeneous_world_coordinates)
        if self.arena is not None:
            model_center = np.asarray(
                self.arena.translate_point_to_world(camera, model_center.reshape((3, 1)))).reshape(-1)

        self.points[camera].append(camera_positions)
        self.centers[camera].append(model_center)
        self.calculate_head_direction(camera, img, rotation_matrix, translation_vector, matrix, dist, extrinsic_matrix)#camera, matrix, dist, rotation_matrix, translation_vector, extrinsic_matrix, img)

    #def calculate_head_direction(self, camera, matrix, dist, r, t, extrinsic, img):
    def calculate_head_direction(self, camera, img, r, t, matrix, dist, extrinsic):
        model_center_projected_point_world = np.asarray(self.calculate_world_coordinates(camera, extrinsic, PROJECTED_MODEL_CENTER)).reshape(-1)
        head_direction_point_world = np.asarray(self.calculate_world_coordinates(camera, extrinsic, HEAD_DIRECTION_POINT)).reshape(-1)
        vector = Vector.from_points(model_center_projected_point_world, head_direction_point_world)
        self.directions[camera].append(vector.unit())
        if img is None:
            return

        projected, _ = cv.projectPoints(np.array([PROJECTED_MODEL_CENTER, HEAD_DIRECTION_POINT]), r, t, matrix, dist)
        draw_direction(projected, img)

    def save_3d_points(self, exp_name, output_path):
        outfile = path.join(output_path, f"{exp_name}_points_{datetime.now().strftime('%m%d%Y%H%M')}.csv")
        print('save points to:', outfile)
        with open(outfile, 'w', newline='\n') as csvfile:
            main_fieldnames = ['FL', 'FR', 'BL', 'BR', 'center', 'direction']
            minor_fieldnames = ['x', 'y', 'z']
            fieldnames = [main + minor for main in main_fieldnames for minor in minor_fieldnames]
            fieldnames = [camera.value + field for field in fieldnames for camera in Camera]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            n_points = max(
                len(self.points[Camera.TOP]),
                len(self.points[Camera.BACK]),
                len(self.points[Camera.LEFT]),
                len(self.points[Camera.RIGHT])
            )
            cameras = [c for c in Camera if len(self.points[c]) > 0]
            for i in range(n_points):
                row = {}
                for camera in cameras:
                    if i > len(self.points[camera]) - 1:
                        continue
                    for j in range(len(self.points[camera][i])):
                        if in_arena(self.points[camera][i][j]):
                            for k in range(len(minor_fieldnames)):
                                row[camera.value + main_fieldnames[j] + minor_fieldnames[k]] = str(self.points[camera][i][j][k])
                    if in_arena(self.centers[camera][i]):
                        for k in range(len(minor_fieldnames)):
                            row[camera.value + main_fieldnames[len(main_fieldnames) - 2] + minor_fieldnames[k]] = str(self.centers[camera][i][k])
                    if len(self.directions[camera]) < i or len(self.directions[camera][i]) == 0:
                        continue
                    for k in range(len(minor_fieldnames)):
                        row[camera.value + main_fieldnames[len(main_fieldnames) - 1] + minor_fieldnames[k]] = str(self.directions[camera][i][k])

                writer.writerow(row)

    def get_world_to_pixels_translation(self, camera, points):
        matrix, dist = matrices[camera]
        image_points = np.array(points, dtype="double")
        (success, rotation_vector, translation_vector) = cv.solvePnP(self.model_points, image_points,
                                                                     matrix, dist)
        if not success:
            print('failed solving pnp')
            return

        rotation_matrix, _ = cv.Rodrigues(rotation_vector)

        def points_to_pixels(p):
            p, _ = cv.projectPoints(p, rotation_vector, translation_vector, matrix, dist)
            return p

        return points_to_pixels, rotation_matrix, translation_vector

