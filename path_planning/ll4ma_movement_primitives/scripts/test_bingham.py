import os
import numpy as np
from oct2py import octave

octave.addpath(os.path.expanduser("~/source_code/bingham/matlab"))

# generate random data
a = np.random.random_sample((10,4))
a /= np.linalg.norm(a, axis=1)[:,None]

octave.bingham_fit(a)
