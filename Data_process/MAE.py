import pyreadr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

def load_json_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def transform_to_image1(data):
    # Initialize the matrix with 100.0 (indicating unused slots)
    matrix = np.full((8, 9), 100.0)
    indices = [
        (0, 3), (0, 4), (0, 5), (0, 6),
        (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
        (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8),
        (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6),(3, 7), (3, 8),
        (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7),(4, 8),
        (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8),
        (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7),
        (7, 3), (7, 4), (7, 5), (7, 6)
    ]
    # Fill the specified indices with data from the input array
    for idx, (i, j) in enumerate(indices):
        matrix[i, j] = data[idx]
    masked_matrix = np.where(matrix == 100, np.nan, matrix)
    return masked_matrix

def transform_to_image2(data):
    # Initialize the matrix with 100.0 (indicating unused slots)
    matrix = np.full((8, 9), 100.0)
    indices = [
        (0, 3), (0, 4), (0, 5), (0, 6),
        (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
        (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8),
        (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 8),
        (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 8),
        (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8),
        (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7),
        (7, 3), (7, 4), (7, 5), (7, 6)
    ]
    # Fill the specified indices with data from the input array
    for idx, (i, j) in enumerate(indices):
        matrix[i, j] = data[idx]
    masked_matrix = np.where(matrix == 100, np.nan, matrix)
    return masked_matrix

def find_td_seq(data, patient_id, eye, age):
    # Check if the patient_id exists in the data
    if patient_id not in data['data']:
        print(f"Patient ID {patient_id} not found.")
        return None

    patient_data = data['data'][patient_id]
    if eye not in patient_data:
        print(f"Eye {eye} not found for patient ID {patient_id}.")
        return None

    visits = patient_data[eye]
    for visit in visits:
        if 'age' in visit and round(visit['age'], 2) == round(age, 2):
            return visit.get('td_seq')

    print(f"No visit found for patient ID {patient_id}, eye {eye}, and age {age}.")
    return None


# def plot_images(td_seq_values, reconstructed_samples, transform_to_image):
#     num_samples = len(td_seq_values)
#
#     fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))
#
#     for i in range(num_samples):
#         # Transform td_seq and reconstruction to image format
#         td_image = transform_to_image(td_seq_values[i])
#         reconstructed_image = transform_to_image(reconstructed_samples[i])
#
#         # Plot the td_seq image
#         axes[i, 0].imshow(td_image, cmap="gray",  interpolation='none')
#         axes[i, 0].set_title(f"Original TD Sequence {i + 1}")
#         axes[i, 0].axis("off")
#
#         # Plot the reconstructed image
#         axes[i, 1].imshow(reconstructed_image, cmap="gray",  interpolation='none')
#         axes[i, 1].set_title(f"Reconstructed Sample {i + 1}")
#         axes[i, 1].axis("off")
#
#     plt.tight_layout()
#     plt.show()

def plot_images(td_seq_values, reconstructed_samples, transform_to_image1,transform_to_image2):
    num_samples = len(td_seq_values)

    fig, axes = plt.subplots(1, num_samples * 2, figsize=(5 * num_samples, 5))

    for i in range(num_samples):
        # Transform td_seq and reconstruction to image format
        td_image = transform_to_image1(td_seq_values[i])
        reconstructed_image = transform_to_image2(reconstructed_samples[i])

        # Plot the td_seq image
        axes[2 * i].imshow(td_image, cmap="gray", interpolation='none')
        axes[2 * i].set_title(f"Original TD Sequence {i + 1}")
        axes[2 * i].axis("off")

        # Plot the reconstructed image
        axes[2 * i + 1].imshow(reconstructed_image, cmap="gray", interpolation='none')
        axes[2 * i + 1].set_title(f"Reconstructed Sample {i + 1}")
        axes[2 * i + 1].axis("off")

    plt.tight_layout()
    plt.show()

# Load and transpose the archetypes matrix
archetypes = pd.read_csv("/Users/xingrobert/Documents/2024/glaucoma progression/VF-diffusion/R/at17_matrix.csv")
archetypes = archetypes.T

coefficients_list = [
    np.array([
        0.452614630266776, 0.0143667622530954, 0.147864933645662, 0.0232907755301355,
        0.0734027314664873, 0.0609583325498919, 0.0271323487395144, 0, 0,
        0.0302928512189371, 0.0220931746976982, 0.112316873065263, 0, 0, 0,
        0.0285712469165719, 0.00709558623478632
    ]),
    np.array([
        0.453913520847353, 0.0649731797537615, 0.104723301327546, 0.0271502848806031,
        0.117049518067677, 0, 0.0458430937056068, 0, 0.0465037700018607, 0,
        0.0090387764981864, 0.0779454481416694, 0, 0.0528602422387338, 0, 0, 0
    ]),
    np.array([
        0.348000696582763, 0.1293146518817, 0.0123799460184007, 0.0179871622092178,
        0.108593900687043, 0, 0.0900109116551368, 0.0809015102207978, 0.0936101876273064,
        0.0548300889745878, 0, 0.0428223068708239, 0, 0.0206985274673571, 0, 0,
        0.000848782621613201
    ])
]


# List to store the reconstruction results
reconstructed_samples = []

for coeff in coefficients_list:
    reconstructed_sample = np.dot(archetypes.values, coeff)
    reconstructed_samples.append(reconstructed_sample)

# for i, sample in enumerate(reconstructed_samples, 1):
#     print(f"Reconstructed Sample {i}:")
#     print(sample)

json_file = './alldata_pro.json'
data = load_json_data(json_file)

search_params = [
    ('647', 'L', 52.7967),
    ('647', 'L', 53.8234),
    ('647', 'L', 54.8856)
]

td_seq_values = []
for patient_id, eye, age in search_params:
    td_seq = find_td_seq(data, patient_id, eye, age)
    td_seq_values.append(td_seq)

print("List of td_seq values:", td_seq_values)
plot_images(td_seq_values, reconstructed_samples, transform_to_image1,transform_to_image2)