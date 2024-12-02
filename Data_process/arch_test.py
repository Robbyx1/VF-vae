import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
import glo_var
from hvf_dataset import HVFDataset as glo
import math
import os

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
    patient_id = str(patient_id)
    if patient_id not in data['data']:
        print(f"Patient ID {patient_id} not found.")
        return None
    patient_data = data['data'][patient_id]
    if eye not in patient_data:
        print(f"Eye {eye} not found for patient ID {patient_id}.")
        return None
    for visit in patient_data[eye]:
        if 'age' in visit and round(visit['age'], 3) == round(age, 3):
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


def denormalize_td_seq(normalized_td_seq):
    # Convert to tensor if needed and get min and max values from glo_var
    min_values = torch.tensor(glo_var.archetype_min, dtype=torch.float32)
    max_values = torch.tensor(glo_var.archetype_max, dtype=torch.float32)

    # Denormalize
    denormalized_td_seq = normalized_td_seq * (max_values - min_values) + min_values
    return denormalized_td_seq

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
        # Skip if either array is None
        if arr1 is None or arr2 is None:
            print("Warning: Encountered None value in one of the arrays, skipping this pair.")
            continue

        # Convert arr1 and arr2 to tensors, ensuring they are numeric arrays
        arr1 = torch.tensor(arr1, dtype=torch.float32) if not isinstance(arr1, torch.Tensor) else arr1
        arr2 = np.array(arr2, dtype=np.float32)  # Ensure arr2 is a numeric array
        arr2 = torch.tensor(arr2)  # Convert arr2 to tensor

        # Calculate the Mean Absolute Error (MAE)
        mae = torch.mean(torch.abs(arr1 - arr2))

        # Calculate mean absolute value of the original data (arr1)
        mean_abs_original = torch.mean(torch.abs(arr1))

        # Calculate percentage error
        percentage_error = (mae / mean_abs_original) * 100
        percentage_errors.append(percentage_error.item())

    return percentage_errors



def plot_images(td_seq_values, reconstructed_samples, max_cols=4, save_filename="plot_output.png"):
    num_samples = len(td_seq_values)
    num_cols = min(max_cols, num_samples * 2)  # Maximum number of columns (pairs per row)
    num_rows = math.ceil((num_samples * 2) / num_cols)  # Calculate required rows

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * max_cols, 5 * num_rows))
    axes = axes.flatten()  # Flatten the axes array for easier indexing

    for i in range(num_samples):
        # Original TD Sequence
        td_image = transform_to_image2(td_seq_values[i])
        axes[2 * i].imshow(td_image, cmap="gray", vmin=0, vmax=1)
        axes[2 * i].set_title(f"Original TD Sequence {i + 1}")
        axes[2 * i].axis("off")

        # Reconstructed Sample
        reconstructed_image = transform_to_image2(reconstructed_samples[i])
        axes[2 * i + 1].imshow(reconstructed_image, cmap="gray", vmin=0, vmax=1)
        axes[2 * i + 1].set_title(f"Reconstructed Sample {i + 1}")
        axes[2 * i + 1].axis("off")

    # Hide any unused axes
    for j in range(2 * num_samples, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    # Save the figure in the current directory
    save_path = os.path.join(os.getcwd(), save_filename)
    plt.savefig(save_path)
    plt.close(fig)  # Close the figure to free up memory

    print(f"Plot saved to {save_path}")


def get_high_decomposed_values(data_dec, data, threshold=0.80):
    td_seq_list = []
    unique_entries = set()

    # Filter data_dec for Type == 1 and HighestDecomposedValue > threshold
    filtered_rows = data_dec[(data_dec['Type'] == 1) & (data_dec['HighestDecomposedValue'] > threshold)]

    for _, row in filtered_rows.iterrows():
        patient_id = row['PatientID']
        eye = row['Eye']
        age = row['Age']

        # Avoid duplicates by using a unique combination of PatientID, Eye, and Age
        if (patient_id, eye, age) not in unique_entries:
            td_seq = find_td_seq(data, patient_id, eye, age)
            if td_seq is not None:
                td_seq_list.append(td_seq)
                unique_entries.add((patient_id, eye, age))  # Track this entry as seen

                # Print details for debugging
                # print(f"Appended td_seq for PatientID: {patient_id}, Eye: {eye}, Age: {age}")
                # print(f"Length of td_seq: {len(td_seq)}")
            else:
                print(f"No matching td_seq found for PatientID: {patient_id}, Eye: {eye}, Age: {age}")
        else:
            print(f"Duplicate found and skipped for PatientID: {patient_id}, Eye: {eye}, Age: {age}")

    return td_seq_list


# def plot_archetype_with_range(min_range, max_range, first_archetype, title="First Archetype vs. Min-Max Range at Each Point", save_filename=None):
#
#     num_points = len(min_range)
#     x = np.arange(num_points)
#
#     # Plot the min and max range as boundaries
#     plt.fill_between(x, min_range, max_range, color="lightgray", label="Range (Min-Max)")
#
#     # Plot the first archetype values
#     plt.plot(x, first_archetype, color="blue", marker="o", label="First Archetype")
#
#     # Add labels, title, and legend
#     plt.xlabel("Point Index")
#     plt.ylabel("Value")
#     plt.title(title)
#     plt.legend()
#
#     # Save or display the plot
#     if save_filename:
#         save_path = os.path.join(os.getcwd(), save_filename)
#         plt.savefig(save_path)
#         plt.close()  # Close the plot to free up memory
#         print(f"Plot saved to {save_path}")
#     else:
#         plt.show()

def plot_archetypes_with_range(min_range, max_range, first_archetype, sixth_archetype, title="Archetypes Comparison with Min-Max Range", save_filename=None):

    num_points = len(min_range)
    x = np.arange(num_points)

    # Plot the min and max range as boundaries
    plt.fill_between(x, min_range, max_range, color="lightgray", label="Range (Min-Max)")

    # Plot the first archetype values
    plt.plot(x, first_archetype, color="blue", marker="o", label="Archetype1")

    # Plot the sixth archetype values
    plt.plot(x, sixth_archetype, color="red", marker="x", linestyle="--", label="Archetype6")

    # Add labels, title, and legend
    plt.xlabel("52 VF td Points")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()

    if save_filename:
        save_path = os.path.join(os.getcwd(), save_filename)
        plt.savefig(save_path)
        plt.close()  # Close the plot to free up memory
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

# Example usage:
# plot_archetypes_with_range(min_range, max_range, first_archetype, sixth_archetype, save_filename="archetype_comparison_plot.png")


# Main function
def main():
    json_file = './alldata_pro.json'
    data = load_json_data(json_file)
    glo_cal = glo('../src/uwhvf/alldata.json')

    file_path = "/Users/xingrobert/Documents/2024/glaucoma progression/VF-diffusion/R/patient_data_decomposed_all_uwhvf.csv"
    data_dec = pd.read_csv(file_path)
    # print(data_dec.columns)
    # search_params = [('1930', 'L', 74.992), ('6205', 'R', 78.324), ('3136', 'L', 76.345)]
    # td_seq_values = [find_td_seq(data, patient_id, eye, age) for patient_id, eye, age in search_params]

    # Load archetypes and calculate reconstructions
    archetypes = pd.read_csv("/Users/xingrobert/Documents/2024/glaucoma progression/VF-diffusion/R/at17_matrix.csv").T
    print(archetypes.shape)
    min_values = archetypes.min(axis=1)  # Minimum value for each row (52 positions)
    max_values = archetypes.max(axis=1)

    min_values = min_values.to_numpy()
    max_values = max_values.to_numpy()

    glo_var.archetype_min = min_values
    glo_var.archetype_max = max_values

    coefficients_list = []
    search_params = []
    for archetype_type in range(1, 18):
        # Filter rows of the current archetype type and find the row with the highest HighestDecomposedValue
        archetype_data = data_dec[data_dec['Type'] == archetype_type]
        if not archetype_data.empty:
            # Find the row with the highest HighestDecomposedValue
            highest_row = archetype_data.loc[archetype_data['HighestDecomposedValue'].idxmax()]

            # Extract the coefficient array from Archetype_1 to Archetype_17 columns
            coefficients = highest_row[['Archetype_1', 'Archetype_2', 'Archetype_3', 'Archetype_4', 'Archetype_5',
                                        'Archetype_6', 'Archetype_7', 'Archetype_8', 'Archetype_9', 'Archetype_10',
                                        'Archetype_11', 'Archetype_12', 'Archetype_13', 'Archetype_14', 'Archetype_15',
                                        'Archetype_16', 'Archetype_17']].values
            coefficients_list.append(coefficients)

            # Collect the PatientID, Eye, and Age to form the search parameters
            patient_id = highest_row['PatientID']
            eye = highest_row['Eye']
            age = highest_row['Age']
            hi_val = highest_row['HighestDecomposedValue']
            search_params.append((patient_id, eye, age))
            # print("test111"+ str(hi_val))
            print(
                f"tttttttt Archetype {archetype_type} - PatientID: {patient_id}, Eye: {eye}, Age: {age}, HighestDecomposedValue: {hi_val}")
        else:
            print(f"No data found for archetype type {archetype_type}. Skipping this type.")

        # Collect the PatientID, Eye, and Age to form the search parameters
        patient_id = highest_row['PatientID']
        eye = highest_row['Eye']
        age = highest_row['Age']
        search_params.append((patient_id, eye, age))


    td_seq_values = []
    unique_entries = set()
    for patient_id, eye, age in search_params:
        if (patient_id, eye, age) not in unique_entries:
            td_seq = find_td_seq(data, patient_id, eye, age)

            if td_seq is None:
                print(f"No matching td_seq found for PatientID: {patient_id}, Eye: {eye}, Age: {age}")
            else:
                td_seq_values.append(td_seq)
                unique_entries.add((patient_id, eye, age))  # Mark this entry as seen

                # Print details for debugging
                print(f"Appended td_seq for PatientID: {patient_id}, Eye: {eye}, Age: {age}")
                print(f"Length of td_seq: {len(td_seq)}")
                # print(f"td_seq: {td_seq}")
        else:
            print(f"Duplicate found and skipped for PatientID: {patient_id}, Eye: {eye}, Age: {age}")

    reconstructed_samples = [archetypes.values @ coeff for coeff in coefficients_list]
    normalized_reconstructed_samples = [
        normalize_reconstruction(sample) for sample in reconstructed_samples
    ]
    #
    # # Calculate MAE and plot results
    mae_results = calculate_mae(td_seq_values, normalized_reconstructed_samples)
    # for i, mae in enumerate(mae_results):
    #     print(f"MAE for pair {i + 1}: {mae}")

    maeP_results = calculate_percentage_mae(td_seq_values, normalized_reconstructed_samples)

    for i, mae in enumerate(maeP_results):
        print(f"MAE Percentage for pair {i + 1}: {mae}")
    if len(td_seq_values) != len(normalized_reconstructed_samples):
        print("Warning: td_seq_values and normalized_reconstructed_samples have different lengths.")
        print(f"td_seq_values length: {len(td_seq_values)}")
        print(f"normalized_reconstructed_samples length: {len(normalized_reconstructed_samples)}")
    # plot_images(td_seq_values, normalized_reconstructed_samples)
    high_decomposed_td_seqs = get_high_decomposed_values(data_dec, data, threshold=0.90)
    denormalized_td_seqs = [denormalize_td_seq(td_seq) for td_seq in high_decomposed_td_seqs]
    print(f"Number : {len(denormalized_td_seqs)}")
    # print("First few high decomposed entries:")
    # for i, td_seq in enumerate(denormalized_td_seqs[:5]):
    #     print(f"Entry {i + 1}: {td_seq}")
    denormalized_array = np.stack([td_seq.numpy() for td_seq in denormalized_td_seqs])
    min_range = denormalized_array.min(axis=0)
    max_range = denormalized_array.max(axis=0)


    # archetypes = pd.read_csv("/Users/xingrobert/Documents/2024/glaucoma progression/VF-diffusion/R/at17_matrix.csv").T
    first_archetype = archetypes.iloc[:, 0].to_numpy()
    sixth_archetype = archetypes.iloc[:, 5].to_numpy()
    print(first_archetype)
    # plot_archetype_with_range(min_range, max_range, first_archetype, save_filename="archetype_plot.png")
    plot_archetypes_with_range(min_range, max_range, first_archetype, sixth_archetype,
                               save_filename="archetype_comparison_plot.png")

    archetype_columns = [f"Archetype_{i}" for i in range(1, 18)]

    # # Calculate the sum of archetype values for each row
    # data_dec['Archetype_Sum'] = data_dec[archetype_columns].sum(axis=1)
    #
    # # Check if each row sum is approximately 1 (allowing for minor floating-point error)
    # tolerance = 1e-4
    # incorrect_rows = data_dec[~np.isclose(data_dec['Archetype_Sum'], 1, atol=tolerance)]
    #
    # # Print the result
    # if incorrect_rows.empty:
    #     print("All rows' Archetype values add up to 1.")
    # else:
    #     print("Rows with Archetype sums not equal to 1:")
    #     print(incorrect_rows[["PatientID", "Eye", "Age", "Archetype_Sum"]])
# Run main
if __name__ == "__main__":
    main()
