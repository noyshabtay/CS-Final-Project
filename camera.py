from enum import Enum


class Camera(Enum):
    TOP = 'top'
    LEFT = 'left'
    RIGHT = 'right'
    BACK = 'back'


def file_to_camera(file):
    if file.find(Camera.TOP.value) != -1:
        return Camera.TOP
    elif file.find(Camera.BACK.value) != -1:
        return Camera.BACK
    elif file.find(Camera.LEFT.value) != -1:
        return Camera.LEFT
    elif file.find(Camera.RIGHT.value) != -1:
        return Camera.RIGHT
