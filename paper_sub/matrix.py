# import torch
# from hvf_dataset import HVFDataset
# import archetypes as arch
# import numpy as np
# import pandas as pd
#
# def perform_archetypal_analysis(json_file, n_archetypes):
#     # Load the dataset
#     dataset = HVFDataset(json_file)
#     data_matrix = torch.stack(dataset.sequences).numpy()
#
#     # Initialize the archetypal analysis model
#     aa = arch.AA(n_archetypes=n_archetypes, verbose=True)
#
#     # Fit the archetypes to the data (no transformation here)
#     aa.fit(data_matrix)
#
#     # Print archetypes
#     print(f"Archetypes (n={n_archetypes}):\n{aa.archetypes_}")
#
#     return aa
#
# if __name__ == "__main__":
#     json_file_path = '../src/uwhvf/alldata.json'
#     help(arch.AA)
#     # Perform archetypal analysis
#     archetypes_model = perform_archetypal_analysis(json_file_path, n_archetypes=17)
#
#     # Save archetypes to a CSV file
#     archetypes_df = pd.DataFrame(archetypes_model.archetypes_)
#     archetypes_df.to_csv("archetypes_n.csv", index=False)
#
#     print("Archetypes saved to 'archetypes.csv'.")

import torch
from hvf_dataset import HVFDataset
import archetypes as arch
import numpy as np
import pandas as pd

def normalize_data(data_matrix):
    """
    Normalize the data matrix using z-score normalization (mean and standard deviation).

    Parameters:
    - data_matrix: The raw data matrix (numpy array).

    Returns:
    - normalized_data: The normalized data matrix.
    - mean: Mean values of each feature (for reversing the normalization).
    - std: Standard deviation values of each feature (for reversing the normalization).
    """
    mean = data_matrix.mean(axis=0)  # Mean of each feature (column)
    std = data_matrix.std(axis=0)   # Std of each feature (column)
    std[std == 0] = 1e-8  # Avoid division by zero
    normalized_data = (data_matrix - mean) / std
    return normalized_data, mean, std

def reverse_normalization(normalized_archetypes, mean, std):
    """
    Reverse z-score normalization to bring data back to the original scale.

    Parameters:
    - normalized_archetypes: Archetypes in normalized scale (numpy array).
    - mean: Mean values of each feature (used for normalization).
    - std: Standard deviation values of each feature (used for normalization).

    Returns:
    - original_archetypes: Archetypes in the original scale.
    """
    original_archetypes = (normalized_archetypes * std) + mean
    return original_archetypes

def perform_archetypal_analysis(json_file, n_archetypes):
    # Load the dataset
    dataset = HVFDataset(json_file)
    data_matrix = torch.stack(dataset.sequences).numpy()

    # Normalize the data using z-score normalization
    print("Applying z-score normalization...")
    normalized_data, mean, std = normalize_data(data_matrix)
    print("Mean of normalized data (should be close to 0):", normalized_data.mean(axis=0))
    print("Std of normalized data (should be close to 1):", normalized_data.std(axis=0))
    print("Min value after normalization:", normalized_data.min())
    print("Max value after normalization:", normalized_data.max())

    # Initialize the archetypal analysis model
    # aa = arch.AA(n_archetypes=n_archetypes, verbose=True)
    aa = arch.AA(n_archetypes=n_archetypes, n_init=1, max_iter=500, tol=1e-4, verbose=True)
    # Fit the archetypes to the normalized data
    aa.fit(normalized_data)

    # Reverse normalization to bring archetypes back to the original scale
    original_scale_archetypes = reverse_normalization(aa.archetypes_, mean, std)

    # # Print archetypes
    # print(f"Archetypes in normalized scale (n={n_archetypes}):\n{aa.archetypes_}")
    # print(f"Archetypes in original scale (n={n_archetypes}):\n{original_scale_archetypes}")

    return original_scale_archetypes

if __name__ == "__main__":
    json_file_path = '../src/uwhvf/alldata.json'

    # Perform archetypal analysis with z-score normalization
    archetypes_original_scale = perform_archetypal_analysis(json_file_path, n_archetypes=17)

    # Save archetypes in the original scale to a CSV file
    archetypes_df = pd.DataFrame(archetypes_original_scale)
    archetypes_df.to_csv("archetypes_original_scale.csv", index=False)

    print("Archetypes saved to 'archetypes_original_scale.csv'.")
