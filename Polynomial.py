import numpy as np

class Polynomial:
    def __init__(self, coefficients):
        self.coefficients = coefficients

    @property
    def degree(self):
        return len(self.coefficients) - 1

    @staticmethod
    def Lagrangian(points):
        a = []
        b = []
        degree = len(points) - 1
        for point in points:
            datapoint = []
            for power in range(degree + 1):
                datapoint.append(point[0]**power)
            a.append(datapoint)
            b.append([point[1]])
        coefficients = np.linalg.solve(a, b)
        return Polynomial([coefficient[0] for coefficient in coefficients])

    def __repr__(self):
        terms = []
        power = 0
        for coefficient in self.coefficients:
            terms.append(str(coefficient) + " * x ^ " + str(power))
            power = power + 1
        return " + ".join(terms)

    def value_at(self, x):
        total = 0
        power = 0
        for coefficient in self.coefficients:
            total = total + coefficient * (x**power)
            power = power + 1
        return total

    @staticmethod
    def regress(x_coords, y_coords, degree):
        inverted_coefficients = np.polyfit(x_coords, y_coords, degree)
        return Polynomial(inverted_coefficients[::-1])

    def __add__(self, other):
        min_degree = 0
        max_degree = 0
        if self.degree > other.degree:
            min_degree = other.degree
            max_degree = self.degree
        else:
            min_degree = self.degree
            max_degree = other.degree
        coords = []
        for i in range(max_degree + 1):
            if i <= min_degree:
                coords.append(self.coefficients[i] + other.coefficients[i])
            elif self.degree > min_degree:
                coords.append(self.coefficients[i])
            else:
                coords.append(other.coefficients[i])
        return Polynomial(coords)
