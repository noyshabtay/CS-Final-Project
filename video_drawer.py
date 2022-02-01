import cv2 as cv


def draw_direction(points, img):
    p1 = points[0].flatten()
    p2 = points[1].flatten()
    p1 = (int(p1[0]), int(p1[1]))
    p2 = (int(p2[0]), int(p2[1]))
    cv.circle(img, p1, 2, (0, 255, 0), 1)
    cv.line(img, p1, p2, (0, 0, 255), 1)


def draw_model(points, img, color = (0,255,0)):
    centers = [(int(p[0]), int(p[1])) for p in points]
    cv.line(img, centers[0], centers[1], color)
    cv.line(img, centers[1], centers[3], color)
    cv.line(img, centers[3], centers[2], color)
    cv.line(img, centers[2], centers[0], color)
