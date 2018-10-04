import tensorflow as tf
import random
import NLib as nl
import numpy as np
from CanvasUtils import CanvasUtils
from Polynomial import Polynomial


class DataBufferer:
    def __init__(self, coordinate_encoding_points, data_encoding_points, coordinate_encoding_length, degree):
        self.coordinate_encoding_points = coordinate_encoding_points
        self.data_encoding_points = data_encoding_points
        self.coordinate_encoding_length = coordinate_encoding_length
        self.degree = degree
        self.x_coords = []
        for x in range(degree + 1):
            self.x_coords.append(x)

        for x_index in range(data_encoding_points):
            x = x_index * degree / (data_encoding_points - 1)
            self.x_coords.append(x)

        coordinate_encoding_start = degree - coordinate_encoding_length

        for x_index in range(coordinate_encoding_points):
            x = coordinate_encoding_start + x_index * coordinate_encoding_length / (coordinate_encoding_points - 1)
            self.x_coords.append(x)

    def encode(self, data):
        points = []
        x = 0
        for y in data:
            points.append([x, y])
            x = x + 1

        points.append([x, 0])
        first_poly = Polynomial.Lagrangian(points)

        x_coords = []
        y_coords = []
        pure_y_coords = []
        min_y = 0

        for point in points:
            x_coords.append(point[0])
            pure_y_coords.append(point[1])

        for x_index in range(self.data_encoding_points):
            x = x_index * self.degree / (self.data_encoding_points - 1)
            y = first_poly.value_at(x)
            x_coords.append(x)
            pure_y_coords.append(y)

        coordinate_encoding_start = self.degree - self.coordinate_encoding_length

        for x_index in range(self.coordinate_encoding_points):
            x = coordinate_encoding_start + x_index * self.coordinate_encoding_length / (self.coordinate_encoding_points - 1)
            y = first_poly.value_at(x)
            x_coords.append(x)
            pure_y_coords.append(y)

        min_y = 0

        for i in range(len(pure_y_coords)):
            poly_y = (pure_y_coords[i] - min_y)
            # y_coords.append(poly_y * random.gauss(1, scatter))
            y_coords.append(poly_y)

        return y_coords

    def decode(self, encoded_data):
        extrapolated_poly = Polynomial.regress(self.x_coords, encoded_data, self.degree)

        extrapolated_y = []
        # zero_corrector = extrapolated_poly.value_at(degree)
        zero_corrector = 0
        for x in range(self.degree):
            y = extrapolated_poly.value_at(x) - zero_corrector
            extrapolated_y.append(y)
        return extrapolated_y


