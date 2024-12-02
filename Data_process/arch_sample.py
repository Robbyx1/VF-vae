import pandas as pd
import numpy as np
import archetypes as arch


def load_archetype_matrix(filepath):
    """
    Load the archetype matrix from a CSV file.

    Args:
        filepath (str): Path to the CSV file containing the archetype matrix.

    Returns:
        np.ndarray: Archetype matrix as a NumPy array.
    """
    archetypes = pd.read_csv(filepath)
    return archetypes.to_numpy()


def normalize_data(values, data_min, data_max, archetype_min, archetype_max):
    """
    Normalize the data to the range of archetypes.

    Args:
        values (np.ndarray): Original data values.
        data_min (float): Minimum value in the data.
        data_max (float): Maximum value in the data.
        archetype_min (float): Minimum value in the archetypes.
        archetype_max (float): Maximum value in the archetypes.

    Returns:
        np.ndarray: Normalized values.
    """
    normalized = (values - data_min) / (data_max - data_min)
    scaled = normalized * (archetype_max - archetype_min) + archetype_min
    return scaled


def decompose_hvf_data(aa, hvf_data):
    """
    Perform decomposition using the archetype analysis model.

    Args:
        aa (arch.AA): Trained archetype analysis model.
        hvf_data (np.ndarray): HVF data to be decomposed.

    Returns:
        np.ndarray: Decomposition coefficients.
    """
    return aa.transform(hvf_data)


def reconstruct_from_coefficients(archetype_matrix, coefficients):
    """
    Reconstruct data using archetype coefficients.

    Args:
        archetype_matrix (np.ndarray): Matrix of archetypes.
        coefficients (np.ndarray): Decomposition coefficients.

    Returns:
        np.ndarray: Reconstructed data.
    """
    return archetype_matrix.T @ coefficients


def main():
    # Load archetype matrix
    archetype_matrix_path = "path/to/archetype_matrix.csv"  # Replace with the actual path
    archetype_matrix = load_archetype_matrix(archetype_matrix_path)

    # Initialize archetypes model
    n_archetypes = 17
    aa = arch.AA(n_archetypes=n_archetypes)
    aa.archetypes_ = archetype_matrix

    # Example data (replace with your actual data)
    hvf_data = np.array([[...]])
    data_min, data_max = -37.69, 22.69  # Replace with actual min/max values in your data
    archetype_min, archetype_max = np.min(archetype_matrix), np.max(archetype_matrix)

    # Normalize the data
    normalized_data = normalize_data(hvf_data, data_min, data_max, archetype_min, archetype_max)

    # Decompose the normalized data
    decomposition_coefficients = decompose_hvf_data(aa, normalized_data)

    # Reconstruct the data from decomposition coefficients
    reconstructed_data = reconstruct_from_coefficients(archetype_matrix, decomposition_coefficients[0])

    # Print results for verification
    print("Decomposition Coefficients:", decomposition_coefficients)
    print("Reconstructed Data:", reconstructed_data)


if __name__ == "__main__":
    main()
