from Controller import Controller
import numpy
from CanvasUtils import CanvasUtils
import NLib

con = Controller()
inputs, outputs, screen = con.make_orig_preserved_batch(1)
red_screen = numpy.array([[row[0] for row in column] for column in screen[0][0][0]], int)
transform_matrix = CanvasUtils.make_transform_from_softmaxed(outputs[0])
transposed_screen = CanvasUtils.transform_by_matrix(screen[0][0][0], transform_matrix)
red_transposed = numpy.array([[row[0] for row in column] for column in transposed_screen], int)
red_transposed_orig = numpy.array([[row[0] for row in column] for column in screen[0][0][1]], int)
corrected_screen = CanvasUtils.transform_by_matrix(transposed_screen, numpy.linalg.inv(transform_matrix))
red_corrected = numpy.array([[row[0] for row in column] for column in corrected_screen], int)
inputs_2, corrected_screens = con.make_next_batch(screen, outputs)
input_to_compare = numpy.moveaxis(inputs[0], 3, 0)[0][0]
red_corrected = numpy.array([[row[0] for row in column] for column in corrected_screens[0]], int)
corrected_proc = NLib.preprocess(corrected_screens[0], 30, 30)
dif = numpy.moveaxis(inputs_2, 3, 0)[2][0] - numpy.moveaxis(inputs[0], 3, 0)[0][0]
end_val = dif
