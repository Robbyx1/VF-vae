
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_hvf(original, reconstruction, epoch, batch_index, results_dir):
    def transform_to_image(data):
        # Initialize the matrix with 100.0 (indicating unused slots)
        matrix = np.full((8, 9), 100.0)
        indices = [
            (0, 3), (0, 4), (0, 5), (0, 6),
            (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
            (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8),
            (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8),
            (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8),
            (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8),
            (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7),
            (7, 3), (7, 4), (7, 5), (7, 6)
        ]
        # Fill the specified indices with data from the input array
        for idx, (i, j) in enumerate(indices):
            matrix[i, j] = data[idx]
        # Apply masking for the value 100, setting it to NaN for visualization purposes
        masked_matrix = np.where(matrix == 100, np.nan, matrix)
        return masked_matrix

    # Transform both original and reconstructed data
    original_img = transform_to_image(original)
    reconstructed_img = transform_to_image(reconstruction)

    # Create a plot with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'Epoch {epoch}, Batch {batch_index}')

    # Plotting original data
    axs[0].imshow(original_img, cmap='gray', interpolation='none', vmin=np.nanmin(original_img),
                  vmax=np.nanmax(original_img))
    axs[0].set_title('Original')
    axs[0].axis('off')  # Hide axes

    # Plotting reconstructed data
    axs[1].imshow(reconstructed_img, cmap='gray', interpolation='none', vmin=np.nanmin(reconstructed_img),
                  vmax=np.nanmax(reconstructed_img))
    axs[1].set_title('Reconstructed')
    axs[1].axis('off')  # Hide axes

    # Save the figure
    plt.savefig(os.path.join(results_dir, f'comparison_epoch{epoch}_batch{batch_index}.png'))
    plt.close()


if __name__ == "__main__":
    test_data_original = np.random.normal(loc=-2.0, scale=5.0, size=54)
    test_data_reconstructed = np.random.normal(loc=-2.0, scale=5.0, size=54)

    # The epoch number and batch index for testing purposes
    epoch = 1
    batch_index = 0

    results_dir = './results_plottest'
    os.makedirs(results_dir, exist_ok=True)

    # Call the plot function
    plot_hvf(test_data_original, test_data_reconstructed, epoch, batch_index, results_dir)

    print("Plot has been generated and saved successfully.")
