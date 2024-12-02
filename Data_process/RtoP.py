import pandas as pd
import numpy as np
from scipy.optimize import nnls
from numpy.linalg import lstsq
import archetypes as arch

# Load the archetype matrix (shape: 52 x 17)
archetypes = pd.read_csv("/Users/xingrobert/Documents/2024/glaucoma progression/VF-diffusion/R/at17_matrix.csv")
archetype_matrix = archetypes.to_numpy()

aa = arch.AA(n_archetypes=17)
aa.archetypes_ = archetype_matrix
# aa.fit(archetype_matrix)
print(aa.archetypes_.shape)


hvf_data = np.array([-6.02, -1.83, -2.28, -2.63, -2.69, -4.21, -2.24, -2.54, -7.17, -3.66,
                   -5.5, -4.19, -3.91, -2.88, -6.07, -5.22, -6.89, -6.51, -4.6, -4.75,
                   -3.75, -4.08, -4.21, -5.72, -6.24, 25.0, -5.93, -8.17, -6.1, -4.31,
                   -2.85, -2.69, -5.83, -4.83, 0.0, -3.87, -4.43, -4.6, -5.57, -5.04,
                   -7.95, -6.28, -6.4, -6.53, -2.74, -4.93, -1.37, -6.15, -4.94, -5.61,
                   -3.93, -4.39, -4.09, -4.72])

td_seq = np.concatenate((hvf_data[:25], hvf_data[26:34], hvf_data[35:]))
td_seq_reshaped = td_seq.reshape(1, 52)

td_seq_transformed = aa.transform(td_seq_reshaped)
print("Decomposition Coefficients:", td_seq_transformed)
print("Sum of Coefficients:", np.sum(td_seq_transformed))
