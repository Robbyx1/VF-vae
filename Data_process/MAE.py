# import pyreadr
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
import glo_var
from hvf_dataset import HVFDataset as glo

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
    matrix = np.pad(matrix, ((2, 2), (2, 1)), mode='constant', constant_values=100.0)
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
    matrix = np.pad(matrix, ((2, 2), (2, 1)), mode='constant', constant_values=100.0)
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


def normalize_to_0_1(td_data):
    valid_min = glo_var.global_min[(static_mask == 1) & (glo_var.global_min != 100.0)]
    valid_max = glo_var.global_max[(static_mask == 1) & (glo_var.global_max != 100.0)]

    min_val = valid_min.min()
    max_val = valid_max.max()

    normalized_data = (td_data - min_val) / (max_val - min_val)

    # Restore the original padding values of 100
    normalized_data = torch.where((static_mask == 1) & (td_data != 100.0), normalized_data, td_data)

    return normalized_data

def plot_images(td_seq_values, reconstructed_samples, transform_to_image1,transform_to_image2cd):
    num_samples = len(td_seq_values)

    fig, axes = plt.subplots(1, num_samples * 2, figsize=(5 * num_samples, 5))

    for i in range(num_samples):
        # Transform td_seq and reconstruction to image format
        td_image = transform_to_image1(td_seq_values[i])
        reconstructed_image = transform_to_image2(reconstructed_samples[i])

        min_matrix = glo_var.global_min[(static_mask == 1) & (glo_var.global_min != 100.0)].min().numpy()
        min_matrix = transform_to_image1(min_matrix)
        max_matrix = glo_var.global_max[(static_mask == 1) & (glo_var.global_max != 100.0)].max().numpy()
        max_matrix = transform_to_image1(max_matrix)

        td_scaled = (td_image - min_matrix) / (max_matrix - min_matrix)
        recon_scaled = (reconstructed_image - min_matrix) / (max_matrix - min_matrix)

        # Plot the td_seq image
        axes[2 * i].imshow(td_image, cmap="gray", interpolation='none', vmin=0, vmax=1)
        axes[2 * i].set_title(f"Original TD Sequence {i + 1}")
        axes[2 * i].axis("off")

        # Plot the reconstructed image
        axes[2 * i + 1].imshow(reconstructed_image, cmap="gray", interpolation='none', vmin=0, vmax=1)
        axes[2 * i + 1].set_title(f"Reconstructed Sample {i + 1}")
        axes[2 * i + 1].axis("off")

    plt.tight_layout()
    plt.show()


def calculate_mae(td_seq_values, reconstructed_samples):
    mae_values = []

    # Iterate over each pair of corresponding matrices
    for td_seq, reconstructed in zip(td_seq_values, reconstructed_samples):
        td_seq_array = np.array(td_seq)
        td_seq_array = np.delete(td_seq_array, [26, 35])
        reconstructed_array = np.array(reconstructed)
        # Flatten the matrices to 1D arrays and compute the absolute error, ignoring NaNs
        diff = np.abs(td_seq_array.flatten() - reconstructed_array.flatten())
        mae = np.nanmean(diff)  # Calculate the mean absolute error, ignoring NaNs
        mae_values.append(mae)

    return mae_values
# Load and transpose the archetypes matrix
archetypes = pd.read_csv("/Users/xingrobert/Documents/2024/glaucoma progression/VF-diffusion/R/at17_matrix.csv")
archetypes = archetypes.T

# coefficients_list = [
#     np.array([
#         0.452614630266776, 0.0143667622530954, 0.147864933645662, 0.0232907755301355,
#         0.0734027314664873, 0.0609583325498919, 0.0271323487395144, 0, 0,
#         0.0302928512189371, 0.0220931746976982, 0.112316873065263, 0, 0, 0,
#         0.0285712469165719, 0.00709558623478632
#     ]),
#     np.array([
#         0.453913520847353, 0.0649731797537615, 0.104723301327546, 0.0271502848806031,
#         0.117049518067677, 0, 0.0458430937056068, 0, 0.0465037700018607, 0,
#         0.0090387764981864, 0.0779454481416694, 0, 0.0528602422387338, 0, 0, 0
#     ]),
#     np.array([
#         0.348000696582763, 0.1293146518817, 0.0123799460184007, 0.0179871622092178,
#         0.108593900687043, 0, 0.0900109116551368, 0.0809015102207978, 0.0936101876273064,
#         0.0548300889745878, 0, 0.0428223068708239, 0, 0.0206985274673571, 0, 0,
#         0.000848782621613201
#     ])
# ]

coefficients_list = [
    np.array([
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
    ])
]


# List to store the reconstruction results
reconstructed_samples = []

for coeff in coefficients_list:
    # reconstructed_sample = np.dot(archetypes.values, coeff)
    reconstructed_sample = archetypes.values @ coeff
    print(coeff.shape)
    reconstructed_samples.append(reconstructed_sample)

# for i, sample in enumerate(reconstructed_samples, 1):
#     print(f"Reconstructed Sample {i}:")
#     print(sample)

json_file = './alldata_pro.json'
data = load_json_data(json_file)

# search_params = [
#     ('647', 'L', 52.7967),
#     ('647', 'L', 53.8234),
#     ('647', 'L', 54.8856)
# ]
search_params = [
    ('1930', 'L', 74.992),
    ('6205', 'R', 78.324),
    ('3136', 'L', 76.345)
]

td_seq_values = []
for patient_id, eye, age in search_params:
    td_seq = find_td_seq(data, patient_id, eye, age)
    td_seq_values.append(td_seq)

# print("List of td_seq values:", td_seq_values)
glo_cal = glo('../src/uwhvf/alldata.json')
res = calculate_mae(td_seq_values, reconstructed_samples)
print(res)
# plot_images(td_seq_values, reconstructed_samples, transform_to_image1,transform_to_image2)


# num_archetypes = archetypes.T.shape[0]
# print(num_archetypes)
# # fig, axes = plt.subplots(1, num_archetypes, figsize=(20, 5))
# fig, axes = plt.subplots(3, 6, figsize=(15, 10))
#
# for i in range(num_archetypes):
#     row = i // 6
#     col = i % 6
#     archetype_vf = transform_to_image2(archetypes.T.iloc[i].values)
#
#     ax = axes[row, col]
#     im = ax.imshow(archetype_vf, cmap="gray", interpolation="none", vmin=np.nanmin(archetype_vf), vmax=np.nanmax(archetype_vf))
#     ax.set_title(f"Archetype {i + 1}")
#     ax.axis("off")
#
# # Add a color bar to the last subplot to show the value scale
# fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.05, pad=0.04)
#
# plt.tight_layout()
# plt.show()