import numpy as np
from scipy.spatial.transform import Rotation

def npmat2euler(mats, seq='zyx', is_degrees=True):
    eulers = []
    for i in range(mats.shape[0]):
        r = Rotation.from_dcm(mats[i])
        eulers.append(r.as_euler(seq, degrees=is_degrees))
    return np.asarray(eulers, dtype='float32')