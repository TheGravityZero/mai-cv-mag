import cv2
import numpy as np


def find_road_number(image: np.ndarray) -> int:
    """
    Найти номер дороги, на которой нет препятсвия в конце пути.

    :param image: исходное изображение
    :return: номер дороги, на котором нет препятсвия на дороге
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # бинарная маска для выделения препятствий, где белый цвет - препятствие
    mask = cv2.inRange(hsv_image, np.array([0, 120, 70]), np.array([10, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w, _ = image.shape
    # Количество дорог равно количеству контуров + 1
    total_routes = len(contours) + 1
    # Ширина дороги равна ширине изображения, деленной на количество дорог
    route_width = w // total_routes

    has_obstacle = [0] * total_routes

    for contour in contours:
        # Выделяем границы контура
        x, _, w, h = cv2.boundingRect(contour)
        # Определяем номер дороги по координате x
        path_id = x // route_width
        if path_id < total_routes:
            has_obstacle[path_id] = 1

    for road_number in range(total_routes):
        if has_obstacle[road_number] == 0:
            return road_number
    return -1
