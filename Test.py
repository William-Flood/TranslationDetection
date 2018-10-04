from DataBufferer import  DataBufferer
import numpy as np
import random
import cv2

def make_matrix(size, sigma):
    orig_points_list = [[0, 0],
                        [0, size],
                        [size, size],
                        [size, 0]]
    transposed_points = []
    for point in orig_points_list:
        x_t = int(random.gauss(point[0], sigma))
        y_t = int(random.gauss(point[1], sigma))
        transposed_points.append([x_t, y_t])
    orig_points = np.float32(orig_points_list)
    transposed_points = np.float32(transposed_points)
    return cv2.getPerspectiveTransform(orig_points, transposed_points)


iterations = 100000
scatter = .05
size = 100
sigma = 10
data_encoding_points = 17
coordinate_encoding_points = 5
coordinate_encoding_length = 1.5
degree = 9
bufferer = DataBufferer(data_encoding_points, coordinate_encoding_points, coordinate_encoding_length, degree)
scattered_first_error = 0
scattered_encoded_error = 0


for i in range(iterations):
    orig_labels = make_matrix(size, sigma)
    first_y = []
    scattered_first_y = []

    for col in orig_labels:
        for cell in col:
            first_y.append(cell)
            scattered_first_y.append(cell * random.gauss(1, scatter))

    encoded_data = bufferer.encode(first_y)

    scattered_encoded_data = []

    for data_elem in encoded_data:
        scattered_encoded_data.append(data_elem * random.gauss(1, scatter))

    decoded_data = bufferer.decode(scattered_encoded_data)

    for j in range(len(first_y)):
        scattered_first_error = scattered_first_error + (scattered_first_y[j] - first_y[j]) ** 2
        scattered_encoded_error = scattered_encoded_error + (decoded_data[j] - first_y[j]) ** 2

    if(i + 1) % (iterations / 10) == 0:
        print(i + 1)

scattered_first_error = scattered_first_error / iterations
scattered_encoded_error = scattered_encoded_error / iterations
print(scattered_encoded_error)
print(scattered_first_error)
