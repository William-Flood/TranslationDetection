import numpy
import scipy.misc as misc
import cv2


class CanvasUtils:
    @staticmethod
    def matrix_multiply(left, right):
        result = []
        for result_column_index in range(len(right)):
            result_column = []
            for result_row_index in range(len(left[0])):
                result_column.append(0)
            result.append(result_column)
        for left_row in range(0, len(left[0])):
            for right_column in range(0, len(right)):
                total = 0
                for left_column in range(0, len(left)):
                    total = total + left[left_column][left_row] * right[right_column][left_column]
                result[right_column][left_row] = total
        return result

    @staticmethod
    def add_point_partial(image_array, color, point_x, point_y, opacity):
        if 0 > point_x or point_x >= len(image_array):
            return
        if 0 > point_y or point_y >= len(image_array[0]):
            return
        partial_adding_color = []
        for color_part in color:
            partial_adding_color.append(color_part * opacity)
        partial_original_color = []
        for color_part in image_array[point_x][point_y]:
            partial_original_color.append(color_part * (1 - opacity))
        for color_index in range(0, len(partial_adding_color)):
            if 255 < int(partial_adding_color[color_index] + partial_original_color[color_index]):
                raise Exception("Math error")
            image_array[point_x][point_y][color_index] = int(
                partial_adding_color[color_index] + partial_original_color[color_index])

    @staticmethod
    def add_point_partials(image_array, color, point_x, point_y):
        floored_x = int(numpy.floor(point_x))
        floored_y = int(numpy.floor(point_y))
        if 0 <= floored_x < len(image_array) and 0 <= floored_y < len(image_array[0]):
            image_array[floored_x][floored_y] = color

    @staticmethod
    def is_between(x_1, x_2, x_mid):
        return (x_1 <= x_2 and x_1 <= x_mid <= x_2) or (x_1 >= x_2 and x_1 >= x_mid >= x_2)

    @staticmethod
    def interpolate_x_of_y(x_1, y_1, x_2, y_2, y_row):
        if y_1 == y_2:
            return x_1
        x_per_y = (x_2 - x_1) / (y_2 - y_1)
        return (y_row - y_1) * x_per_y + x_1

    @staticmethod
    def draw_triangle_row(image_array, color, x_1, y_1, x_2, y_2, x_3, y_3, y_row):
        x_2_3 = CanvasUtils.interpolate_x_of_y(x_2, y_2, x_3, y_3, y_row)
        x_1_2 = CanvasUtils.interpolate_x_of_y(x_1, y_1, x_2, y_2, y_row)
        x_3_1 = CanvasUtils.interpolate_x_of_y(x_3, y_3, x_1, y_1, y_row)

        x_left = 0
        x_right = 0

        if CanvasUtils.is_between(x_1, x_2, x_1_2):
            x_left = x_1_2
            x_right = x_1_2
        elif CanvasUtils.is_between(x_2, x_3, x_2_3):
            x_left = x_2_3
            x_right = x_2_3
        elif CanvasUtils.is_between(x_3, x_1, x_3_1):
            x_left = x_3_1
            x_right = x_3_1

        if CanvasUtils.is_between(x_2, x_3, x_2_3):
            if x_2_3 > x_right:
                x_right = x_2_3
            else:
                x_left = x_2_3

        if CanvasUtils.is_between(x_3, x_1, x_3_1):
            if x_3_1 > x_right:
                x_right = x_3_1
            elif x_3_1 < x_left:
                x_left = x_3_1

        CanvasUtils.add_point_partials(image_array, color, x_left, y_row)
        x_scan = int(x_left) + 1

        while x_scan < x_right:
            CanvasUtils.add_point_partials(image_array, color, x_scan, y_row)
            x_scan = x_scan + 1

        CanvasUtils.add_point_partials(image_array, color, x_right, y_row)

    @staticmethod
    def add_triangle(image_array, color, x_1, y_1, x_2, y_2, x_3, y_3):
        y_top = y_1
        y_bottom = y_1

        if y_2 > y_1:
            y_bottom = y_2
        else:
            y_top = y_2

        if y_3 > y_bottom:
            y_bottom = y_3
        elif y_top > y_3:
            y_top = y_3

        CanvasUtils.draw_triangle_row(image_array, color, x_1, y_1, x_2, y_2, x_3, y_3, y_top)
        y_scan = int(y_top) + 1
        while y_scan < y_bottom:
            CanvasUtils.draw_triangle_row(image_array, color, x_1, y_1, x_2, y_2, x_3, y_3, y_scan)
            y_scan = y_scan + 1
        CanvasUtils.draw_triangle_row(image_array, color, x_1, y_1, x_2, y_2, x_3, y_3, y_bottom)

    @staticmethod
    def transposeByPerspective(original_array, orig_points, transformed_points):
        M = cv2.getPerspectiveTransform(orig_points, transformed_points)
        transposed_array = cv2.warpPerspective(original_array, M, original_array.shape[:-1], cv2.INTER_LINEAR,
                                               cv2.BORDER_CONSTANT, borderValue=(255,255,255))
        return transposed_array, M

    @staticmethod
    def transformByMatrix(original_array, matrix):
        transposed_array = cv2.warpPerspective(original_array, matrix, original_array.shape[:-1], cv2.INTER_LINEAR,
                                               cv2.BORDER_CONSTANT, (255,255,255))
        return transposed_array

    @staticmethod
    def in_range(x_1, x_2, radius):
        if x_1 - radius <= x_2 <= x_1 + radius:
            return True
        else:
            return False
