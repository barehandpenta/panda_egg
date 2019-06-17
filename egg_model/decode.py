import numpy as np
# Scale ndarray to value 0-1 and flatten it
def decode_inputs(array):
    array = np.asfarray(array)
    array[:, :, 0] = array[:, :, 0] / 480
    array[:, :, 1] = array[:, :, 1] / 640
    return array.flatten()