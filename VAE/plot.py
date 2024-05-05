
import matplotlib.pyplot as plt
import numpy as np
import os


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

def plot_hvf(original, reconstruction, epoch, batch_index, results_dir):
    # def transform_to_image(data):
    #     # Initialize the matrix with 100.0 (indicating unused slots)
    #     matrix = np.full((8, 9), 100.0)
    #     indices = [
    #         (0, 3), (0, 4), (0, 5), (0, 6),
    #         (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
    #         (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8),
    #         (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8),
    #         (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8),
    #         (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8),
    #         (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7),
    #         (7, 3), (7, 4), (7, 5), (7, 6)
    #     ]
    #     # Fill the specified indices with data from the input array
    #     for idx, (i, j) in enumerate(indices):
    #         matrix[i, j] = data[idx]
    #     # Apply masking for the value 100, setting it to NaN for visualization purposes
    #     masked_matrix = np.where(matrix == 100, np.nan, matrix)
    #     return masked_matrix

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
    plt.savefig(os.path.join(results_dir, f'comparison_epoch{epoch}_batch{batch_index}_ori.png'))
    plt.close()

def plot_all_reconstructions(originals, reconstructions, epoch, results_dir):
    num_cases = len(originals)
    num_epochs = len(reconstructions[0])
    fig, axes = plt.subplots(num_cases, num_epochs + 1, figsize=(22, 20))  # Plus one for the original

    for i in range(num_cases):
        for e in range(num_epochs):
            recon_img = transform_to_image(reconstructions[i][e])
            axes[i, e].imshow(recon_img, cmap='gray', interpolation='none')
            axes[i, e].axis('off')

        # Plot original in the last column
        orig_img = transform_to_image(originals[i])
        axes[i, num_epochs].imshow(orig_img, cmap='gray', interpolation='none')
        axes[i, num_epochs].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'comparison_grid_epoch{epoch}.png'))
    plt.close()


if __name__ == "__main__":
    test_data_original = np.random.normal(loc=-2.0, scale=5.0, size=54)
    test_data_reconstructed = np.random.normal(loc=-2.0, scale=5.0, size=54)
    original = [-6.02, -1.83, -2.28, -2.63, -2.69, -4.21, -2.24, -2.54, -7.17, -3.66, -5.5, -4.19, -3.91, -2.88, -6.07,
                -5.22, -6.89, -6.51, -4.6, -4.75, -3.75, -4.08, -4.21, -5.72, -6.24, 25.0, -5.93, -8.17, -6.1, -4.31,
                -2.85, -2.69, -5.83, -4.83, 0.0, -3.87, -4.43, -4.6, -5.57, -5.04, -7.95, -6.28, -6.4, -6.53, -2.74,
                -4.93, -1.37, -6.15, -4.94, -5.61, -3.93, -4.39, -4.09, -4.72]
    original2 =  [-9.72, -8.91, -12.63, -9.36, -12.62, -5.08, -8.69, -7.02, -7.89, -8.06, -10.38, -6.12, -7.1, -8.93, -12.09, -8.33, -6.52, -5.76, -14.73, -8.64, -8.73, -12.3, -6.75, -8.76, -12.73, 21.0, -5.73, -14.88, -10.08, -11.89, -7.11, -6.11, -6.65, -7.27, 0.0, -6.93, -9.09, -5.23, -5.59, -5.34, -7.14, -7.27, -5.86, -7.79, -5.12, -3.95, -6.26, -7.68, -4.68, -5.52, -8.35, -4.11, -5.95, -7.56]

    # The epoch number and batch index for testing purposes
    epoch = 1
    batch_index = 0

    results_dir = './results_plottest'
    os.makedirs(results_dir, exist_ok=True)

    # Call the plot function
    # plot_hvf(test_data_original, test_data_reconstructed, epoch, batch_index, results_dir)
    plot_hvf(original, original2, epoch, batch_index, results_dir)

    print("Plot has been generated and saved successfully.")
