import json
import matplotlib.pyplot as plt
from collections import Counter
import os
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np


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

# Function to classify MD based on Hodapp-Parrish-Anderson classification
def classify_md(md_value):
    if md_value < -12:
        return "severe"
    elif -12 <= md_value <= -6:
        return "moderate"
    else:
        return "mild"


# Load the dataset
def load_json_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


# Process the dataset and classify MD values
# Process the dataset and classify MD values
def process_data(data):
    classification_counts = {"mild": [], "moderate": [], "severe": []}
    lost_cases_count = 0
    for patient_id, patient_data in data['data'].items():
        for key, visits in patient_data.items():
            if key in ['L', 'R']:  # Only process the visits for "L" and "R" keys
                if isinstance(visits, list):
                    for visit in visits:
                        if isinstance(visit, dict):
                            md_value = visit.get('MD', None)
                            decomposed_values = visit.get('DecomposedValues', None)  # Assuming it's a list
                            td_seq = visit.get('td_seq', None)  # Assuming td_seq is present
                            age = visit.get('age', None)

                            if md_value is not None and decomposed_values is not None:
                                stage = classify_md(md_value)

                                # Find the index of the highest value (Type) and second-highest value
                                highest_index = decomposed_values.index(max(decomposed_values)) + 1
                                sorted_values = sorted(enumerate(decomposed_values, 1), key=lambda x: x[1],
                                                       reverse=True)
                                second_highest_index, second_highest_value = sorted_values[1]

                                modified_decomposed_values = decomposed_values.copy()
                                modified_decomposed_values[highest_index - 1] = 0

                                # Append the information as a dictionary
                                classification_counts[stage].append({
                                    "Type": highest_index,
                                    "SecondHighestIndex": second_highest_index,
                                    "SecondHighestValue": second_highest_value,
                                    "TD_Seq": td_seq,
                                    "Age": age,
                                    "MD": md_value,
                                    "ModifiedDecomposedValues": modified_decomposed_values
                                })
                            else:
                                lost_cases_count += 1
                                if md_value is None:
                                    print(f" {patient_id} - {age} - 'MD' value is missing")
                                if decomposed_values is None:
                                    print(f"  {patient_id} - {age} - 'DecomposedValues' are missing")
                        else:
                            print(f"Unexpected visit format: {visit} (Type: {type(visit)})")
                            lost_cases_count += 1

    return classification_counts, lost_cases_count


# Plot the archetype distribution for each stage
def plot_archetype_distribution(classification_counts, total_cases, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for stage, archetypes in classification_counts.items():
        # Extract the 'Type' values for plotting
        type_values = [archetype['Type'] for archetype in archetypes]
        second_highest_indices = [archetype['SecondHighestIndex'] for archetype in archetypes]

        # Count occurrences of each 'Type' and 'SecondHighestIndex'
        type_counter = Counter(type_values)
        second_highest_counter = Counter(second_highest_indices)

        # Sort archetype types numerically for plotting
        sorted_types = sorted(type_counter.keys())
        type_counts = [type_counter[atype] for atype in sorted_types]

        # Sort second-highest indices numerically for plotting
        sorted_second_highest = sorted(second_highest_counter.keys())
        second_highest_counts = [second_highest_counter[atype] for atype in sorted_second_highest]

        # Plot 'Type' distribution
        plt.figure(figsize=(10, 6))
        plt.bar([str(atype) for atype in sorted_types], type_counts, color='skyblue')
        plt.title(f'Archetype Distribution for {stage.capitalize()} Stage (Total: {len(archetypes)})')
        plt.xlabel('Archetype Type (Highest)')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'archetype_distribution_{stage}_highest.png'))
        plt.close()

        # Plot 'SecondHighestIndex' distribution
        plt.figure(figsize=(10, 6))
        plt.bar([str(atype) for atype in sorted_second_highest], second_highest_counts, color='lightcoral')
        plt.title(f'Second-Highest Archetype Distribution for {stage.capitalize()} Stage (Total: {len(archetypes)})')
        plt.xlabel('Second-Highest Archetype Index')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'archetype_distribution_{stage}_second_highest.png'))
        plt.close()

    print(f"Plots saved in directory: {output_dir}")
    print(f"Total cases processed: {total_cases}")


# def plot_detailed_histogram(classification_counts, stage, archetype_type, output_dir):
#     os.makedirs(output_dir, exist_ok=True)
#
#     # Filter samples for the given stage and archetype type
#     samples = [
#         sample for sample in classification_counts[stage]
#         if sample['Type'] == archetype_type
#     ]
#
#     # Extract SecondHighestIndex for the filtered samples
#     second_highest_indices = [sample['SecondHighestIndex'] for sample in samples]
#
#     # Count occurrences of each SecondHighestIndex
#     counter = Counter(second_highest_indices)
#     sorted_indices = sorted(counter.keys())  # Sort indices numerically
#     counts = [counter[idx] for idx in sorted_indices]
#
#     # Plot the histogram
#     plt.figure(figsize=(10, 6))
#     plt.bar([str(idx) for idx in sorted_indices], counts, color='skyblue')
#     plt.title(f'SecondHighestIndex Distribution for Archetype {archetype_type} in {stage.capitalize()} Stage')
#     plt.xlabel('SecondHighestIndex')
#     plt.ylabel('Count')
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
#
#     # Save the plot
#     output_path = os.path.join(output_dir, f'{stage}_archetype_{archetype_type}_second_highest.png')
#     plt.savefig(output_path)
#     plt.close()
#
#     print(f"Plot saved at: {output_path}")

def plot_detailed_histogram(classification_counts, stage, archetype_type, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Filter samples for the given stage and archetype type
    samples = [
        sample for sample in classification_counts[stage]
        if sample['Type'] == archetype_type
    ]

    # Extract SecondHighestIndex for the filtered samples
    second_highest_indices = [sample['SecondHighestIndex'] for sample in samples]

    # Count occurrences of each SecondHighestIndex
    counter = Counter(second_highest_indices)
    sorted_indices = sorted(counter.keys())  # Sort indices numerically
    counts = [counter[idx] for idx in sorted_indices]

    # Calculate total samples and percentages
    total_samples = len(second_highest_indices)
    percentages = [(count / total_samples) * 100 for count in counts]

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    bars = plt.bar([str(idx) for idx in sorted_indices], percentages, color='skyblue')  # Use percentages for height

    # Add count labels above each bar
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 str(count), ha='center', va='bottom', fontsize=10)

    # Add title, labels, and formatting
    plt.title(f'SecondHighestIndex Distribution for Archetype {archetype_type} in {stage.capitalize()} Stage')
    plt.xlabel('Archetype Type')
    plt.ylabel('Percentage (%)')  # Y-axis shows percentage
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save the plot
    output_path = os.path.join(output_dir, f'{stage}_archetype_{archetype_type}_second_highest.png')
    plt.savefig(output_path)
    plt.close()

    print(f"Plot saved at: {output_path}")

def plot_archetype_heatmap(classification_counts, stage, archetype_type, archetypes_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # Load the archetypes matrix
    archetypes = pd.read_csv(archetypes_csv).values  # Load as a numpy array
    archetypes = archetypes

    # Filter the samples for the given stage and archetype type
    samples = [
        sample for sample in classification_counts[stage]
        if sample['Type'] == archetype_type
    ]

    # Initialize a 52-length array for the sum
    total_sum = np.zeros(52)

    # Compute the dot product and accumulate the sum
    for sample in samples:
        modified_values = np.array(sample["ModifiedDecomposedValues"])
        dot_product = np.dot(modified_values, archetypes)
        total_sum += dot_product

    # Transform the sum array to a 2D matrix
    matrix = transform_to_image2(total_sum)

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    # plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    # plt.imshow(matrix, cmap='RdBu', interpolation='nearest', vmin=-np.nanmax(np.abs(matrix)),
    #            vmax=np.nanmax(np.abs(matrix)))
    data_min = np.nanmin(matrix)
    data_max = np.nanmax(matrix)
    plt.imshow(matrix, cmap='RdBu', interpolation='nearest', vmin=data_min, vmax=data_max)
    plt.colorbar(label='Sum of HVF Values')
    plt.title(f'Heatmap for Archetype {archetype_type} in {stage.capitalize()} Stage')
    plt.xlabel('X-Axis')
    plt.ylabel('Y-Axis')

    # Save the plot
    output_path = f"{output_dir}/heatmap_{stage}_archetype_{archetype_type}.png"
    plt.savefig(output_path)
    plt.close()

    print(f"Heatmap saved at: {output_path}")


def plot_archetype_matrices(archetypes_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Load the archetypes matrix
    archetypes = pd.read_csv(archetypes_csv).values  # Load as a numpy array
    # archetypes = archetypes.T  # Transpose to ensure shape is (17, 52)
    n_rows, n_cols = 4, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12))
    fig.suptitle("Archetypes 1 to 16", fontsize=16, y=0.98)

    # Define color map limits
    vmin, vmax = -38, 38  # Set the grayscale range

    for i in range(16):  # Plot only the first 16 archetypes
        ax = axes[i // n_cols, i % n_cols]  # Get the subplot axis
        archetype_vector = archetypes[i]  # Get the i-th archetype (52-length vector)

        # Convert to 2D matrix using the transform_to_image2 function
        matrix = transform_to_image2(archetype_vector)

        # Plot the heatmap for the archetype
        im = ax.imshow(matrix, cmap='RdBu', interpolation='nearest', vmin=vmin, vmax=vmax)
        ax.set_title(f"AT{i + 1}", fontsize=10)
        ax.axis("off")  # Turn off axis labels for a clean look

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Position for the colorbar
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Value Range", fontsize=10)

    output_path = os.path.join(output_dir, "archetypes_16.png")
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to fit the colorbar
    plt.savefig(output_path)
    plt.close()

    print(f"Archetypes heatmap saved at: {output_path}")
def main():
    # json_file = './alldata_pro.json'
    json_file = './alldata_26.json'
    data = load_json_data(json_file)

    classification_counts, lost_cases_count = process_data(data)

    total_cases = sum(len(v) for v in classification_counts.values())
    print(f"Total lost cases due to unexpected format: {lost_cases_count}")
    # output_dir = "archetype_test"
    # plot_archetype_distribution(classification_counts, total_cases, output_dir)

    output_dir = "secondary_histograms"
    plot_detailed_histogram(classification_counts, stage="mild", archetype_type=14, output_dir=output_dir)
    #
    #
    # output_dir = "heatmaps"
    archetypes_csv = "./at17_matrix.csv"
    # plot_archetype_heatmap(classification_counts, stage="mild", archetype_type=9, archetypes_csv=archetypes_csv,
    #                        output_dir=output_dir)
    # output_dir = "archetypes_heatmaps"
    # plot_archetype_matrices(archetypes_csv, output_dir)

if __name__ == "__main__":
    main()