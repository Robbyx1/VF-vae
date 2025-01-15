import pandas as pd
import numpy as np
import archetypes as arch
import json
import os
import matplotlib.pyplot as plt
import math
import torch


def load_archetype_matrix(filepath):
    archetypes = pd.read_csv(filepath)
    return archetypes.to_numpy()

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

def find_top_td_seq_by_type(data):
    top_td_by_type = {}

    # Iterate through each patient in the dataset
    for patient_id, patient_data in data['data'].items():
        # Check for each eye ('L' and 'R') in the patient data
        for eye in ['L', 'R']:
            if eye not in patient_data:
                continue
            visits = patient_data[eye]
            for visit in visits:
                if not isinstance(visit, dict):
                    print(f"Warning: Expected a dictionary for visit data, but got {type(visit)}.")
                    continue

                # Safely access attributes in the visit dictionary
                visit_type = visit.get('Type')
                highest_value = visit.get('HighestDecomposedValue')
                td_seq = visit.get('td_seq')
                td_seq = np.array(td_seq)
                td_seq = np.concatenate((td_seq[:25], td_seq[26:34], td_seq[35:]))

                try:
                    highest_value = float(highest_value)
                except (ValueError, TypeError):
                    print(
                        f"Warning: Invalid HighestDecomposedValue '{highest_value}' for patient {patient_id}, eye {eye}.")
                    continue

                # Skip if any required value is missing
                if visit_type is None or td_seq is None:
                    print(f"Skipping incomplete visit data for patient {patient_id}, eye {eye}.")
                    continue

                # Check if this type has been encountered before
                if visit_type not in top_td_by_type:
                    top_td_by_type[visit_type] = (highest_value, td_seq)
                else:
                    # Update if this visit has a higher 'HighestDecomposedValue' for this type
                    if highest_value > top_td_by_type[visit_type][0]:
                        top_td_by_type[visit_type] = (highest_value, td_seq)

    return top_td_by_type

def process_sorted_results(sorted_results, aa):
    processed_results = []

    for  td_seq in sorted_results:
        td_seq = np.array(td_seq)
        # reshape to 1 row
        hvf_data = td_seq.reshape(1, -1)

        # Perform decomposition using the provided function
        decomposition_result = decompose_hvf_data(aa, hvf_data)
        processed_results.append(decomposition_result[0])

    return processed_results

def reconstruct_from_coefficients(archetype_matrix, coefficients_list):
    archetype_matrix = archetype_matrix.T
    reconstructed_data = [archetype_matrix @ coeff for coeff in coefficients_list]
    return reconstructed_data

def calculate_percentage_mae(flattened_actual, flattened_reconstructed):
    percentage_mae_list = []

    for actual, reconstructed in zip(flattened_actual, flattened_reconstructed):
        mae = np.mean(np.abs(actual - reconstructed))
        mean_actual = np.mean(np.abs(actual))  # Use the absolute mean of actual values
        percentage_mae = (mae / mean_actual) * 100

        # Append the percentage MAE for this data point
        percentage_mae_list.append(percentage_mae)

    return percentage_mae_list


def plot_all_results(flattened_actual, flattened_reconstructed, save_path="reconstruction_comparison.png"):

    n_samples = len(flattened_actual)
    n_cols = 2
    n_rows = (n_samples + 1) // n_cols  # Calculate rows needed for 2 columns

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 3), constrained_layout=True)
    axes = axes.flatten()

    for i in range(n_samples):
        axes[i].plot(flattened_actual[i], label='Original')
        axes[i].plot(flattened_reconstructed[i], label='Reconstructed', linestyle='--')
        axes[i].set_title(f"Type {i+1} - Original vs Reconstructed")
        axes[i].legend()

    # Hide any extra subplots if there are less than 17 samples
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Save the figure
    plt.savefig(save_path)
    print(f"Plot saved as {save_path}")
    # plt.show()

def plot_images(td_seq_values, reconstructed_samples, max_cols=4, save_filename="plot_output.png"):
    num_samples = len(td_seq_values)
    num_cols = min(max_cols, num_samples * 2)  # Maximum number of columns (pairs per row)
    num_rows = math.ceil((num_samples * 2) / num_cols)  # Calculate required rows

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * max_cols, 5 * num_rows))
    axes = axes.flatten()  # Flatten the axes array for easier indexing

    for i in range(num_samples):
        # Original TD Sequence
        td_image = transform_to_image2(td_seq_values[i])
        axes[2 * i].imshow(td_image, cmap="gray",interpolation='none')
        # axes[2 * i].imshow(td_image, cmap="gray", vmin=0, vmax=1)
        axes[2 * i].set_title(f"Original TD Sequence {i + 1}")
        axes[2 * i].axis("off")

        # Reconstructed Sample
        reconstructed_image = transform_to_image2(reconstructed_samples[i])
        # axes[2 * i + 1].imshow(reconstructed_image, cmap="gray", vmin=0, vmax=1)
        axes[2 * i + 1].imshow(reconstructed_image, cmap="gray",interpolation='none')
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

def decompose_hvf_data(aa, hvf_data):
    td_seq_transformed = aa.transform(hvf_data)
    return td_seq_transformed


def process_and_save_all_data(data, archetype_matrix, output_csv_path):
    """
    Processes all data from the dataset, decomposes it using the archetype matrix,
    and saves the results into a CSV file.

    Parameters:
        data (dict): The dataset loaded from JSON.
        archetype_matrix (numpy array): The archetype matrix for decomposition.
        output_csv_path (str): Path to save the output CSV file.
    """
    # Prepare archetypes
    aa = arch.AA(n_archetypes=17)
    aa.archetypes_ = archetype_matrix

    # Prepare output data
    output_rows = []

    # Process each patient
    for patient_id, patient_data in data['data'].items():
        for eye in ['L', 'R']:
            if eye not in patient_data:
                continue
            visits = patient_data[eye]
            for visit in visits:
                if not isinstance(visit, dict):
                    print(f"Warning: Unexpected visit data format for patient {patient_id}, eye {eye}.")
                    continue

                td_seq = visit.get('td_seq')
                # age = visit.get('age')
                age = round(visit.get('age'), 6)
                if td_seq is None:
                    print(f"Skipping missing `td_seq` for patient {patient_id}, eye {eye}.")
                    continue

                # Preprocess `td_seq` as needed (e.g., remove indices)
                td_seq = np.array(td_seq)
                td_seq = np.concatenate((td_seq[:25], td_seq[26:34], td_seq[35:]))

                # Normalize the data
                min_d, max_d = -37.69, 22.69
                min_val_arch = np.min(archetype_matrix)
                max_val_arch = np.max(archetype_matrix)
                td_seq_normalized_mid = (td_seq - min_d) / (max_d - min_d)
                td_seq_normalized = td_seq_normalized_mid * (max_val_arch - min_val_arch) + min_val_arch

                # Reshape for decomposition
                td_seq_reshaped = td_seq_normalized.reshape(1, -1)

                # Perform decomposition
                decomposition_result = decompose_hvf_data(aa, td_seq_reshaped)
                coefficients = decomposition_result[0]

                # Find the highest decomposed value and its type
                highest_value = np.max(coefficients)
                type_index = np.argmax(coefficients) + 1

                # Save the result row
                row = {
                    'PatientID': patient_id,
                    'Eye': eye,
                    'Age': age,
                    'HighestDecomposedValue': highest_value,
                    'Type': type_index,
                }
                # Add decomposition coefficients (17 columns)
                row.update({f'Archetype_{i + 1}': coef for i, coef in enumerate(coefficients)})
                output_rows.append(row)

    # Convert to DataFrame and save to CSV
    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(output_csv_path, index=False)
    print(f"All processed data saved to {output_csv_path}")


def main():
    archetype_matrix = load_archetype_matrix(
        "/Users/xingrobert/Documents/2024/glaucoma progression/VF-diffusion/R/at17_matrix.csv")
    print("Archetype Matrix Shape:", archetype_matrix.shape)
    json_file = './alldata_pro.json'
    data = load_json_data(json_file)
    output_csv_path = "./patient_decomposed_all_python.csv"
    process_and_save_all_data(data, archetype_matrix, output_csv_path)

    # top_td_by_type = find_top_td_seq_by_type(data)
    # sorted_results = dict(sorted(top_td_by_type.items(), key=lambda x: int(x[0])))
    # flatten_values = [np.array(td_seq) for _, (_, td_seq) in sorted_results.items()]
    # min_val_arch = np.min(archetype_matrix)
    # max_val_arch = np.max(archetype_matrix)
    # min_d = -37.69
    # max_d = 22.69
    # print(f"Min value in archetype matrix: {min_val_arch}")
    # print(f"Max value in archetype matrix: {max_val_arch}")
    # print(f"Min value in data: {min_d}")
    # print(f"Max value in data: {max_d}")
    # normalized_values = [(values - min_d) / (max_d - min_d) for values in flatten_values]
    # scaled_values = [
    #     values * (max_val_arch - min_val_arch) + min_val_arch for values in normalized_values
    # ]
    # # for i, normalized in enumerate(normalized_values):
    # #     print(f"Normalized values for set {i}: {normalized}")
    # #     print(f"Range: Min = {np.min(normalized)}, Max = {np.max(normalized)}")
    # flatten_values = scaled_values
    # # for visit_type, (highest_value, td_seq) in sorted_results.items():
    # #     print(f"Type: {visit_type}")
    # #     print(f"  HighestDecomposedValue: {highest_value}")
    # #     print(f"  td_seq: {td_seq}")
    #
    # aa = arch.AA(n_archetypes=17)
    # aa.archetypes_ = archetype_matrix
    # # print("Loaded Archetypes Shape:", aa.archetypes_.shape)
    #
    # processed_results = process_sorted_results(flatten_values, aa)
    # # coefficients_list = [result['DecompositionCoefficients'].flatten() for result in processed_results.values()]
    # coefficients_list = processed_results
    # print("Processed Results:")
    # for i, result in enumerate(processed_results):
    #     print(f"Result {i + 1}: {result}")
    # reconstructed_data = reconstruct_from_coefficients(archetype_matrix, coefficients_list)
    # # for visit_type, reconstructed in zip(processed_results.keys(), reconstructed_data):
    # #     print(f"Type: {visit_type}")
    # #     print(f"  Reconstructed Data: {reconstructed}")
    # mae_results = calculate_percentage_mae(flatten_values, reconstructed_data)
    # for i, mae in enumerate(mae_results):
    #     print(f"MAE for pair {i + 1}: {mae}")
    #
    # plot_all_results(flatten_values, reconstructed_data, save_path="reconstruction_comparison.png")
    # plot_images(flatten_values, reconstructed_data, max_cols=4, save_filename="python_output.png")
    # # plot_images(flatten_values,reconstructed_data)
    # # print("Decomposition Coefficients:", td_seq_transformed)
    # # print("Sum of Coefficients:", np.sum(td_seq_transformed))


if __name__ == "__main__":
    main()
