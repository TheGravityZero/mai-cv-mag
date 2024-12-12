import cv2
import numpy as np

from collections import deque


ACTIONS_LIST = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
]

def find_way_from_maze(image: np.ndarray) -> tuple:
    """
    Найти путь через лабиринт.

    :param image: изображение лабиринта
    :return: координаты пути из лабиринта в виде (x, y), где x и y - это массивы координат
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

    h, w = binary_image.shape

    # Первая белая клетка в первой строке
    begin = (0, np.where(binary_image[0] == 255)[0][0])
    # Последняя белая клетка в последней строке
    end = (h - 1, np.where(binary_image[-1] == 255)[0][0])

    # Алгоритм BFS
    q = deque([begin])
    visited = set([begin])
    prev = dict()
    prev[begin] = None

    while len(q) > 0:
        cur = q.popleft()
        # Если текущая клетка - конечная, то прерываем поиск
        if cur == end:
            break

        for action in ACTIONS_LIST:
            next_action = (cur[0] + action[0], cur[1] + action[1])

            if (0 <= next_action[0] < h) and (0 <= next_action[1] < w) and binary_image[next_action] == 255 and next_action not in visited:
                q.append(next_action)
                visited.add(next_action)
                prev[next_action] = cur

    # Восстановление пути
    path = []
    cur = end
    while cur:
        path.append(cur)
        cur = prev[cur]

    x_path, y_path = zip(*path)
    return np.array(x_path), np.array(y_path)
