
from CanvasUtils import CanvasUtils
import random
import numpy


class GameManager:
    def __init__(self, size, sigma):
        self.size = size
        self.sigma = sigma
        self.colors = []
        self.transposed_colors_1 = []
        self.transposed_colors_2 = []
        self.colors_array = numpy.array([], numpy.float32)
        self.transposed_colors_array = numpy.array([], numpy.float32)
        self.x_target = 0
        self.y_target = 0
        self.transform_matrix_1 = []

    def step(self):
        self.colors = []
        self.transposed_colors_1 = []
        for color_column_index in range(0, self.size):
            self.colors.append([])
            self.transposed_colors_1.append([])
            for color_row_index in range(0, self.size):
                self.colors[-1].append([255, 255, 255])
                self.transposed_colors_1[-1].append([0, 0, 0])
        for triangle_index in range(10):
            x_1 = random.randint(0, self.size)
            y_1 = random.randint(0, self.size)
            x_2 = int(random.gauss(x_1, 10))
            if x_2 < 0:
                x_2 = 0
            if x_2 >= self.size:
                x_2 = self.size - 1
            y_2 = int(random.gauss(y_1, 10))
            if y_2 < 0:
                y_2 = 0
            if y_2 >= self.size:
                y_2 = self.size - 1
            x_3 = int(random.gauss(x_2, 10))
            if x_3 < 0:
                x_3 = 0
            if x_3 >= self.size:
                x_3 = self.size - 1
            y_3 = int(random.gauss(y_2, 10))
            if y_3 < 0:
                y_3 = 0
            if y_3 >= self.size:
                y_3 = self.size - 1
            color = [random.randint(0, 255) for i in range(0, 3)]
            CanvasUtils.add_triangle(self.colors, color, x_1, y_1, x_2, y_2, x_3, y_3)
        orig_points_list = [[0, 0],
                            [0, self.size],
                            [self.size, self.size],
                            [self.size, 0]]
        transposed_points = []
        for point in orig_points_list:
            x_t = int(random.gauss(point[0], self.sigma))
            y_t = int(random.gauss(point[1], self.sigma))
            transposed_points.append([x_t, y_t])

        orig_points = numpy.float32(orig_points_list)
        transposed_points = numpy.float32(transposed_points)
        self.colors_array = numpy.array(self.colors, numpy.uint8)
        self.transposed_colors_1, self.transform_matrix_1 = \
            CanvasUtils.transposeByPerspective(self.colors_array, orig_points, transposed_points)
        transposed_points = []
        for point in orig_points_list:
            x_t = int(random.gauss(point[0], self.sigma))
            y_t = int(random.gauss(point[1], self.sigma))
            transposed_points.append([x_t, y_t])
        transposed_points = numpy.float32(transposed_points)
        self.transposed_colors_2, transform_matrix_2 = \
            CanvasUtils.transposeByPerspective(self.transposed_colors_1, orig_points, transposed_points)
        transform_matrix_2 = numpy.linalg.inv(transform_matrix_2)
        return [self.colors_array, self.transposed_colors_1, self.transposed_colors_2], \
            self.transform_matrix_1, transform_matrix_2

