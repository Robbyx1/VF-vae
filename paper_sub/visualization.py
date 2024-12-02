import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Function to transform each archetype row into a matrix
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
    # Add padding with a constant value of 100.0 and mask those regions
    matrix = np.pad(matrix, ((2, 2), (2, 1)), mode='constant', constant_values=100.0)
    masked_matrix = np.where(matrix == 100, np.nan, matrix)
    return masked_matrix


# Main function to load data, transform, and visualize
def visualize_archetypes(csv_file):
    # Load the archetypes CSV file
    df = pd.read_csv(csv_file)

    # Create a figure for the plots
    fig, axes = plt.subplots(3, 6, figsize=(18, 9))  # Adjust layout as needed
    axes = axes.flatten()  # Flatten for easier indexing

    # Plot each archetype
    for i, row in enumerate(df.values):
        if i >= 17:  # Limit to the first 17 archetypes
            break
        # Transform the row into a matrix
        matrix = transform_to_image2(row)

        # Plot the grayscale image
        ax = axes[i]
        # im = ax.imshow(matrix, cmap='gray', interpolation='nearest')
        # im = ax.imshow(matrix, cmap='gray', interpolation='nearest', vmin=-37.69, vmax=22.69)
        im = ax.imshow(matrix, cmap='gray', interpolation='nearest', vmin=-38, vmax=38)
        ax.set_title(f'Archetype {i + 1}')
        ax.axis('off')

    # Hide unused subplots if fewer than 18
    for j in range(len(df.values), len(axes)):
        axes[j].axis('off')

    # Add a colorbar to the last plot
    fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


# Run the visualization
if __name__ == "__main__":
    csv_file_path = "archetypes_original_scale.csv"
    # csv_file_path = "archetypes.csv"
    visualize_archetypes(csv_file_path)
