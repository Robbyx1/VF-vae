import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
import glo_var
from hvf_dataset import HVFDataset as glo

# Static mask setup
static_mask = torch.tensor([
    [0., 0., 0., 1., 1., 1., 1., 0., 0.],
    [0., 0., 1., 1., 1., 1., 1., 1., 0.],
    [0., 1., 1., 1., 1., 1., 1., 1., 1.],
    [1., 1., 1., 1., 1., 1., 1., 0., 1.],
    [1., 1., 1., 1., 1., 1., 1., 0., 1.],
    [0., 1., 1., 1., 1., 1., 1., 1., 1.],
    [0., 0., 1., 1., 1., 1., 1., 1., 0.],
    [0., 0., 0., 1., 1., 1., 1., 0., 0.]
], dtype=torch.float32)
static_mask = F.pad(static_mask, (2, 1, 2, 2), value=0.0)

# Load JSON data
def load_json_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Transformation functions
def transform_to_image1(data):
    # Set up matrix and indices for transform
    matrix = np.full((8, 9), 100.0)
    indices = [ (0, 3), (0, 4), (0, 5), (0, 6),
        (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
        (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8),
        (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6),(3, 7), (3, 8),
        (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7),(4, 8),
        (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8),
        (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7),
        (7, 3), (7, 4), (7, 5), (7, 6) ]
    for idx, (i, j) in enumerate(indices):
        matrix[i, j] = data[idx]
    matrix = np.pad(matrix, ((2, 2), (2, 1)), mode='constant', constant_values=100.0)
    return np.where(matrix == 100, np.nan, matrix)

def transform_to_image2(data):
    matrix = np.full((8, 9), 100.0)
    indices = [ (0, 3), (0, 4), (0, 5), (0, 6),
        (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
        (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8),
        (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 8),
        (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 8),
        (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8),
        (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7),
        (7, 3), (7, 4), (7, 5), (7, 6) ]
    for idx, (i, j) in enumerate(indices):
        matrix[i, j] = data[idx]
    matrix = np.pad(matrix, ((2, 2), (2, 1)), mode='constant', constant_values=100.0)
    return np.where(matrix == 100, np.nan, matrix)

def find_td_seq(data, patient_id, eye, age):
    if patient_id not in data['data']:
        print(f"Patient ID {patient_id} not found.")
        return None
    patient_data = data['data'][patient_id]
    if eye not in patient_data:
        print(f"Eye {eye} not found for patient ID {patient_id}.")
        return None
    for visit in patient_data[eye]:
        if 'age' in visit and round(visit['age'], 2) == round(age, 2):
            td_seq = visit.get('td_seq')
            return normalize_td_seq(td_seq)
    print(f"No visit found for patient ID {patient_id}, eye {eye}, and age {age}.")
    return None


def normalize_td_seq(td_seq):
    td_seq = torch.tensor(td_seq, dtype=torch.float32)
    td_seq = torch.cat((td_seq[:25], td_seq[26:34], td_seq[35:]))
    normalized_data = (td_seq - glo_var.global_min) / (glo_var.global_max - glo_var.global_min)

    return normalized_data

def normalize_reconstruction(reconstructed_sample):
    reconstructed_sample = np.array(reconstructed_sample)
    min_values = np.array(glo_var.archetype_min)
    max_values = np.array(glo_var.archetype_max)
    normalized_sample = (reconstructed_sample - min_values) / (max_values - min_values)
    return normalized_sample


def calculate_mae(list1, list2):
    mae_values = []
    for arr1, arr2 in zip(list1, list2):
        arr1 = arr1.numpy() if isinstance(arr1, torch.Tensor) else arr1
        arr2 = arr2.numpy() if isinstance(arr2, torch.Tensor) else arr2
        mae = np.mean(np.abs(arr1 - arr2))
        mae_values.append(mae)
    return mae_values

def calculate_percentage_mae(list1, list2):
    percentage_errors = []
    for arr1, arr2 in zip(list1, list2):
        arr1 = torch.tensor(arr1) if not isinstance(arr1, torch.Tensor) else arr1
        arr2 = torch.tensor(arr2) if not isinstance(arr2, torch.Tensor) else arr2
        mae = torch.mean(torch.abs(arr1 - arr2))
        mean_abs_original = torch.mean(torch.abs(arr1))
        percentage_error = (mae / mean_abs_original) * 100
        percentage_errors.append(percentage_error.item())

    return percentage_errors
# Plot images
def plot_images(td_seq_values, reconstructed_samples):
    num_samples = len(td_seq_values)
    fig, axes = plt.subplots(1, num_samples * 2, figsize=(5 * num_samples, 5))
    for i in range(num_samples):
        td_image = transform_to_image2(td_seq_values[i])
        reconstructed_image = transform_to_image2(reconstructed_samples[i])
        axes[2 * i].imshow(td_image, cmap="gray", vmin=0, vmax=1)
        axes[2 * i].set_title(f"Original TD Sequence {i + 1}")
        axes[2 * i].axis("off")
        axes[2 * i + 1].imshow(reconstructed_image, cmap="gray", vmin=0, vmax=1)
        axes[2 * i + 1].set_title(f"Reconstructed Sample {i + 1}")
        axes[2 * i + 1].axis("off")
    plt.tight_layout()
    plt.show()

# Main function
def main():
    json_file = './alldata_pro.json'
    data = load_json_data(json_file)
    glo_cal = glo('../src/uwhvf/alldata.json')
    search_params = [('1930', 'L', 74.992), ('6205', 'R', 78.324), ('3136', 'L', 76.345)]
    td_seq_values = [find_td_seq(data, patient_id, eye, age) for patient_id, eye, age in search_params]

    # Load archetypes and calculate reconstructions
    archetypes = pd.read_csv("/Users/xingrobert/Documents/2024/glaucoma progression/VF-diffusion/R/at17_matrix.csv").T
    print(archetypes.shape)
    min_values = archetypes.min(axis=1)  # Minimum value for each row (52 positions)
    max_values = archetypes.max(axis=1)

    min_values = min_values.to_numpy()
    max_values = max_values.to_numpy()

    glo_var.archetype_min = min_values
    glo_var.archetype_max = max_values

    coefficients_list = [np.array([
        0, 0, 0, 0, 0, 0, 0, 0.826681189333964, 0, 0, 0, 0, 0, 0.138168567304504,
        0.0227139260964307, 0, 0.012561261797865
    ]),
    np.array([
        0, 0, 0.0676982105773679, 0, 0, 0.0700614236550268, 0, 0, 0.112688225514538,
        0, 0.701544123864884, 0, 0, 0, 0.0482021748912421, 0, 0
    ]),
    np.array([
        0, 0, 0, 0, 0, 0.115218224597891, 0.0304424600703449, 0, 0, 0, 0,
        0.796420688707216, 0.0582422823610317, 0, 0, 0, 0
    ])]
    reconstructed_samples = [archetypes.values @ coeff for coeff in coefficients_list]
    normalized_reconstructed_samples = [
        normalize_reconstruction(sample) for sample in reconstructed_samples
    ]

    # Calculate MAE and plot results
    mae_results = calculate_mae(td_seq_values, normalized_reconstructed_samples)
    for i, mae in enumerate(mae_results):
        print(f"MAE for pair {i + 1}: {mae}")

    maeP_results = calculate_percentage_mae(td_seq_values, normalized_reconstructed_samples)
    for i, mae in enumerate(maeP_results):
        print(f"MAE Percentage for pair {i + 1}: {mae}")
    plot_images(td_seq_values, normalized_reconstructed_samples)

# Run main
if __name__ == "__main__":
    main()
