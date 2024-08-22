
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from config import init_mask as mask



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

    # Create a subplot with epochs as rows and cases as columns, plus one row for the original
    fig, axes = plt.subplots(num_epochs + 1, num_cases,
                             figsize=(num_cases * 2, num_epochs * 2 + 2))  # Adjust figsize accordingly

    # Loop over each case
    for j in range(num_cases):
        # Plot each epoch in a separate row
        for i in range(num_epochs):
            recon_img = transform_to_image(reconstructions[j][i])
            if i == 0:  # Set column titles at the top
                axes[i, j].set_title(f'Case {j + 1}')
            axes[i, j].imshow(recon_img, cmap='gray', interpolation='none')
            axes[i, j].axis('off')

        # Plot the original data in the last row for each case
        orig_img = transform_to_image(originals[j])
        axes[num_epochs, j].imshow(orig_img, cmap='gray', interpolation='none')
        axes[num_epochs, j].axis('off')
        axes[num_epochs, j].set_xlabel('Original')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'comparison_grid_epoch{epoch}.png'))
    plt.close()


def plot_samples(sampled_data, results_dir='results_vae_samples'):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    num_samples = len(sampled_data)
    fig, axes = plt.subplots(5, 2, figsize=(10, 25))  # 5 rows and 2 columns for the grid

    for i, ax in enumerate(axes.flatten()):
        if i < num_samples:
            print(sampled_data[i])
            img_data = transform_to_image(sampled_data[i])
            ax.imshow(img_data, cmap='gray', interpolation='none')
            ax.set_title(f'Sample {i + 1}')
            ax.axis('off')
        else:
            ax.axis('off')  # Turn off axis for empty plots

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'vae_samples.png'))
    plt.close()


def visualize_latent_space(model, data_loader, device,  results_dir='latent space'):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    model.eval()
    latents = []
    with torch.no_grad():
        for inputs in data_loader:
            inputs = inputs.to(device)
            mu, logvar = model.encode(inputs.view(-1, 54))  # Adjust shape if necessary
            latents.append(mu.cpu().numpy())

    latents = np.concatenate(latents)

    # Assuming a 2D latent space for visualization
    plt.figure(figsize=(10, 7))
    plt.scatter(latents[:, 0], latents[:, 1], alpha=0.5)
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('Latent Space Distribution')
    plt.grid(True)
    # plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'latent_space.png'))
    plt.close()


def save_loss_plot(train_losses, title, results_dir='loss'):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='MAE + KDL Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(results_dir, 'loss.png'))
    plt.close()
    print("loss printed")

# def plot_comparison(originals, reconstructions, results_dir='results_vae_comparisons'):
#     if not os.path.exists(results_dir):
#         os.makedirs(results_dir)
#     num_samples = min(len(originals), len(reconstructions))
#     fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))  # Two columns: one for original and one for reconstruction
#
#     for i in range(num_samples):
#         # Ensure data is squeezed of any singleton dimensions
#         original_img = np.squeeze(originals[i])
#         reconstruction_img = np.squeeze(reconstructions[i])
#
#         # Plot original data
#         ax = axes[i, 0]
#         ax.imshow(original_img, cmap='gray', interpolation='none')
#         ax.set_title(f'Original {i + 1}')
#         ax.axis('off')
#
#         # Plot reconstructed data
#         ax = axes[i, 1]
#         ax.imshow(reconstruction_img, cmap='gray', interpolation='none')
#         ax.set_title(f'Reconstruction {i + 1}')
#         ax.axis('off')
#
#     plt.tight_layout()
#     plt.savefig(os.path.join(results_dir, 'comparison.png'))
#     plt.close()

def plot_comparison(originals, reconstructions, mask, num_samples=5, results_dir='results_comparison'):
    import os
    import matplotlib.pyplot as plt
    import numpy as np

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    fig, axs = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))  # 2 columns for original and reconstructed
    mask = mask.cpu().numpy()  # Convert mask to numpy array once
    for i in range(num_samples):
        original_img = originals[i].squeeze()
        reconstruction_img = reconstructions[i][-1].squeeze()  # Assuming last epoch's reconstruction

        # Apply the mask
        original_display = np.where(mask == 1, original_img, np.nan)  # Set masked areas to NaN for original
        reconstruction_display = np.where(mask == 1, reconstruction_img, np.nan)  # Set masked areas to NaN for reconstruction

        # Plot original data
        axs[i, 0].imshow(original_display, cmap='gray', interpolation='none', vmin=np.nanmin(original_display), vmax=np.nanmax(original_display))
        axs[i, 0].set_title(f'Original Sample {i+1}')
        axs[i, 0].axis('off')  # Turn off axis

        # Plot reconstructed data
        axs[i, 1].imshow(reconstruction_display, cmap='gray', interpolation='none', vmin=np.nanmin(reconstruction_display), vmax=np.nanmax(reconstruction_display))
        axs[i, 1].imshow(reconstruction_display, cmap='gray_r', interpolation='none', vmin=np.nanmin(reconstruction_display), vmax=np.nanmax(reconstruction_display))

        axs[i, 1].set_title(f'Reconstructed Sample {i+1}')
        axs[i, 1].axis('off')  # Turn off axis

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'comparison.png'))
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