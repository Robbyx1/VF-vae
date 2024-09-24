
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from config import init_mask
import seaborn as sns
import pandas as pd




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


# def save_loss_plot(train_losses, title, results_dir='loss'):
#     if not os.path.exists(results_dir):
#         os.makedirs(results_dir)
#
#     # Create the plot
#     plt.figure(figsize=(10, 6))
#     plt.plot(train_losses, label='MAE + KDL Train Loss')
#     plt.title(title)
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True)
#
#     plt.savefig(os.path.join(results_dir, 'loss.png'))
#     plt.close()
#     print("loss printed")

def save_loss_plot(train_losses, val_losses, title, results_dir='loss'):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='x')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Save the plot to the specified directory
    plt.savefig(os.path.join(results_dir, f'dim {title}.png'))
    plt.close()
    print("Loss comparison plot saved.")



def plot_comparison(originals, reconstructions, mask, title="Comparison of Original and Reconstruction", num_samples=5, results_dir='results_comparison'):

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    fig, axs = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))  # 2 columns for original and reconstructed
    mask = mask.cpu().numpy()  # Convert mask to numpy array once
    for i in range(num_samples):
        original_img = originals[i].squeeze()*mask
        reconstruction_img = reconstructions[i][-1].squeeze()  # last epoch's reconstruction


        original_display = np.where(mask == 1, original_img, np.nan)  # Set masked areas to NaN for original
        # original_display = original_display * std + mean
        reconstruction_display = np.where(mask == 1, reconstruction_img, np.nan)  # Set masked areas to NaN for reconstruction
        # reconstruction_display = reconstruction_display*std + mean
        # Plot original data
        axs[i, 0].imshow(original_display, cmap='gray', interpolation='none', vmin=np.nanmin(original_display), vmax=np.nanmax(original_display))
        axs[i, 0].set_title(f'Original Sample {i+1}')
        axs[i, 0].axis('off')  # Turn off axis

        # Plot reconstructed data
        # axs[i, 1].imshow(reconstruction_display, cmap='gray', interpolation='none', vmin=np.nanmin(reconstruction_display), vmax=np.nanmax(reconstruction_display))
        axs[i, 1].imshow(reconstruction_display, cmap='gray', interpolation='none', vmin=np.nanmin(reconstruction_display), vmax=np.nanmax(reconstruction_display))

        axs[i, 1].set_title(f'Reconstructed Sample {i+1}')
        axs[i, 1].axis('off')  # Turn off axis
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'comparison {title}.png'))
    plt.close()

def plot_single_hvf(image, results_dir, file_name='hvf_plot.png'):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Convert the image tensor to numpy if it's a torch tensor
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(image, torch.Tensor):
        image = image.numpy()

    # Call the init_mask function to generate the mask tensor and convert to numpy
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mask = init_mask(device).cpu().numpy()

    # Apply the mask to the image
    image = image.squeeze()  # Remove unnecessary dimensions
    image_dis = np.where(mask == 1, image, np.nan)  # Apply the mask

    # Debugging output
    print("Mask:\n", mask)
    print("Masked Image Display:\n", image_dis)

    fig, ax = plt.subplots()
    # Plotting the data
    ax.imshow(image_dis, cmap='gray', interpolation='none', vmin=np.nanmin(image_dis), vmax=np.nanmax(image_dis))
    ax.set_title("Visual Field Image")
    ax.axis('off')  # Hide axes

    # Save the figure
    plt.savefig(os.path.join(results_dir, file_name))
    plt.close()

def plot_hvf_arch(samples, results_dir, file_name='hvf_row_plot.png'):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Set up a figure for 10 plots in a row
    fig, axes = plt.subplots(1, len(samples), figsize=(20, 5))  # 10 plots in a row

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mask = init_mask(device).cpu().numpy()

    for idx, sample in enumerate(samples):
        hvf_image = sample['td_data']
        decomposed_value = sample['HighestDecomposedValue']

        # Convert the tensor to numpy
        if isinstance(hvf_image, torch.Tensor):
            hvf_image = hvf_image.cpu().numpy()

        # Apply the mask to the image
        hvf_image = hvf_image.squeeze()  # Remove unnecessary dimensions
        hvf_image_dis = np.where(mask == 1, hvf_image, np.nan)  # Apply the mask

        # Plotting each image
        ax = axes[idx]
        ax.imshow(hvf_image_dis, cmap='gray', interpolation='none', vmin=np.nanmin(hvf_image_dis), vmax=np.nanmax(hvf_image_dis))
        ax.set_title(f"{decomposed_value:.4f}")  # Set the title as the HighestDecomposedValue
        ax.axis('off')  # Hide axes

    # Save the figure
    plt.savefig(os.path.join(results_dir, file_name))
    plt.close()


def visualize_latent_vectors(latent_vectors):
    # Convert latent vectors to a numpy array for easier plotting
    latent_array = np.array([lv.squeeze().numpy() for lv in latent_vectors])  # Shape: [10, latent_dim]

    # Get the number of latent dimensions
    latent_dim = latent_array.shape[1]

    # Set up a figure
    plt.figure(figsize=(10, 6))

    # Plot each dimension of the latent vector across the samples
    for i in range(latent_dim):
        plt.plot(range(1, latent_array.shape[0] + 1), latent_array[:, i], label=f'Latent Dim {i + 1}')

    # Add titles and labels
    plt.title('Latent Vector Evolution')
    plt.xlabel('Latent Vector Index (1-10)')
    plt.ylabel('Latent Value')
    plt.legend(loc='upper right')

    # Show the plot
    plt.grid(True)
    plt.show()


def visualize_latent_distribution(latent_vectors):
    # Convert latent vectors to a numpy array for easier plotting
    latent_array = np.array([lv.squeeze().numpy() for lv in latent_vectors])  # Shape: [10, latent_dim]

    # Set up a figure
    plt.figure(figsize=(10, 6))

    # Create a box plot for each latent dimension
    plt.boxplot(latent_array, vert=True, patch_artist=True,
                labels=[f'Dim {i + 1}' for i in range(latent_array.shape[1])])

    # Add titles and labels
    plt.title('Latent Vector Distribution Across Dimensions')
    plt.xlabel('Latent Dimension')
    plt.ylabel('Latent Value')

    # Show the plot
    plt.grid(True)
    plt.show()

def visualize_latent_distribution_violin(latent_vectors):
    # Convert latent vectors to a pandas DataFrame for easier plotting with seaborn
    latent_array = np.array([lv.squeeze().numpy() for lv in latent_vectors])  # Shape: [10, latent_dim]
    df = pd.DataFrame(latent_array, columns=[f'Dim {i+1}' for i in range(latent_array.shape[1])])

    # Set up a figure
    plt.figure(figsize=(10, 6))

    # Create a violin plot for each latent dimension
    sns.violinplot(data=df)

    # Add titles and labels
    plt.title('Latent Vector Distribution Across Dimensions')
    plt.xlabel('Latent Dimension')
    plt.ylabel('Latent Value')

    # Show the plot
    plt.grid(True)
    plt.show()


def visualize_latent_vectors_across_dimensions(encoded_latent_vectors):
    plt.figure(figsize=(10, 6))

    # Define a color map for distinct types
    colors = plt.cm.get_cmap('tab10', len(encoded_latent_vectors))

    # Loop through each type and plot the mean latent vector for that type
    for idx, (type_val, latent_vectors) in enumerate(encoded_latent_vectors.items()):
        # Convert latent vectors to a numpy array
        latent_array = np.array([lv.squeeze().numpy() for lv in latent_vectors])  # Shape: [5, 10]

        # Compute the mean vector across the samples for each type
        mean_latent_vector = latent_array.mean(axis=0)  # Shape: [latent_dim]

        # Plot the mean latent vector across dimensions
        plt.plot(range(1, len(mean_latent_vector) + 1), mean_latent_vector, label=f'Type {type_val}', color=colors(idx))

    # Add titles and labels
    plt.title('Mean Latent Vector Across Dimensions for Each Type')
    plt.xlabel('Latent Dimension')
    plt.ylabel('Latent Value')
    plt.legend(loc='upper right', ncol=2)  # Add fewer columns to reduce clutter

    # Show the plot
    plt.grid(True)
    plt.show()

def visualize_sorted_latent_vectors(encoded_latent_vectors):
    plt.figure(figsize=(10, 6))

    # Convert keys to integers (if they are stored as strings) and sort them
    sorted_keys = sorted(encoded_latent_vectors.keys(), key=lambda x: int(x))

    # Define a color map with a gradual change, e.g., 'viridis' or 'plasma'
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_keys)))

    # Loop through each type and plot the mean latent vector for that type
    for idx, type_val in enumerate(sorted_keys):
        latent_vectors = encoded_latent_vectors[type_val]
        latent_array = np.array([lv.squeeze().numpy() for lv in latent_vectors])  # Shape: [5, 10]

        # Compute the mean vector across the samples for each type
        mean_latent_vector = latent_array.mean(axis=0)  # Shape: [latent_dim]

        # Plot the mean latent vector across dimensions using the gradual color map
        plt.plot(range(1, len(mean_latent_vector) + 1), mean_latent_vector,
                 label=f'Type {type_val}', color=colors[idx])

    # Add titles and labels
    plt.title('Mean Latent Vector Across Dimensions for Each Type')
    plt.xlabel('Latent Dimension')
    plt.ylabel('Latent Value')
    plt.legend(loc='upper right', ncol=2)  # Adjust legend to fit more compactly
    plt.grid(True)

    # Show the plot
    plt.show()


def visualize_original_latent_vectors(encoded_latent_vectors):
    plt.figure(figsize=(10, 6))

    # Convert keys to integers (if they are stored as strings) and sort them
    sorted_keys = sorted(encoded_latent_vectors.keys(), key=lambda x: int(x))

    # Define a color map with a gradual change, e.g., 'viridis' or 'plasma'
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_keys)))

    # Loop through each type and plot the 5 latent vectors for that type using the same color
    for idx, type_val in enumerate(sorted_keys):
        latent_vectors = encoded_latent_vectors[type_val]
        latent_array = np.array([lv.squeeze().numpy() for lv in latent_vectors])  # Shape: [5, latent_dim]

        # Plot each of the 5 vectors for the current type with the same color
        for i in range(latent_array.shape[0]):
            if i == 0:
                # Add label only for the first vector to avoid duplicate labels
                plt.plot(range(1, latent_array.shape[1] + 1), latent_array[i],
                         label=f'Type {type_val}', color=colors[idx])
            else:
                plt.plot(range(1, latent_array.shape[1] + 1), latent_array[i],
                         color=colors[idx])

    # Add titles and labels
    plt.title('Latent Vectors Across Dimensions for Each Type')
    plt.xlabel('Latent Dimension')
    plt.ylabel('Latent Value')
    plt.legend(loc='upper right', ncol=2)  # Adjust legend to fit more compactly
    plt.grid(True)

    # Show the plot
    plt.show()


def visualize_highlighted_latent_vectors(encoded_latent_vectors):
    plt.figure(figsize=(10, 6))

    # Convert keys to integers (if they are stored as strings) and sort them
    sorted_keys = sorted(encoded_latent_vectors.keys(), key=lambda x: int(x))

    # Define a color map with a gradual change, e.g., 'viridis'
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_keys)))

    # Manually assign special colors for type 1, type 8, and type 13
    special_colors = {1: 'darkred', 8: 'red', 13: 'blue'}

    # Loop through each type and plot the mean latent vector for that type
    for idx, type_val in enumerate(sorted_keys):
        latent_vectors = encoded_latent_vectors[type_val]
        latent_array = np.array([lv.squeeze().numpy() for lv in latent_vectors])  # Shape: [5, 10]

        # Compute the mean vector across the samples for each type
        mean_latent_vector = latent_array.mean(axis=0)  # Shape: [latent_dim]

        # Assign color based on type
        color = special_colors.get(int(type_val), colors[idx])  # Use special color for types 1, 8, and 13

        # Plot the mean latent vector across dimensions
        plt.plot(range(1, len(mean_latent_vector) + 1), mean_latent_vector,
                 label=f'Type {type_val}', color=color)

    # Add titles and labels
    plt.title('Mean Latent Vector Across Dimensions for Each Type')
    plt.xlabel('Latent Dimension')
    plt.ylabel('Latent Value')
    plt.legend(loc='upper right', ncol=2)  # Adjust legend to fit more compactly
    plt.grid(True)

    # Show the plot
    plt.show()

def visualize_highlighted_and_faded_latent_vectors(encoded_latent_vectors):
    plt.figure(figsize=(10, 6))

    # Convert keys to integers (if they are stored as strings) and sort them
    sorted_keys = sorted(encoded_latent_vectors.keys(), key=lambda x: int(x))

    # Define a color map with a gradual change, e.g., 'viridis'
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_keys)))

    # Manually assign special colors for type 1, type 8, and type 13
    special_colors = {1: 'black', 8: 'red', 13: 'blue'}

    # Loop through each type and plot the mean latent vector for that type
    for idx, type_val in enumerate(sorted_keys):
        latent_vectors = encoded_latent_vectors[type_val]
        latent_array = np.array([lv.squeeze().numpy() for lv in latent_vectors])  # Shape: [5, 10]

        # Compute the mean vector across the samples for each type
        mean_latent_vector = latent_array.mean(axis=0)  # Shape: [latent_dim]

        # Assign color based on type and set transparency (alpha) for non-special types
        if int(type_val) in special_colors:
            color = special_colors[int(type_val)]
            alpha = 1.0  # No fading for special types
            linewidth = 2.5  # Make highlighted lines thicker
        else:
            color = colors[idx]
            alpha = 0.3  # Fade other types
            linewidth = 1.0  # Regular thickness for faded lines

        # Plot the mean latent vector across dimensions
        plt.plot(range(1, len(mean_latent_vector) + 1), mean_latent_vector,
                 label=f'Type {type_val}', color=color, alpha=alpha, linewidth=linewidth)

    # Add titles and labels
    plt.title('Mean Latent Vector Across Dimensions for Each Type')
    plt.xlabel('Latent Dimension')
    plt.ylabel('Latent Value')
    plt.legend(loc='upper right', ncol=2)  # Adjust legend to fit more compactly
    plt.grid(True)

    # Show the plot
    plt.show()

def visualize_interpolation(interpolated_results):
    plt.figure(figsize=(15, 5))

    # Plot each interpolated result
    for idx, result in enumerate(interpolated_results):
        plt.subplot(1, len(interpolated_results), idx + 1)
        plt.imshow(result, cmap='gray')
        plt.axis('off')
        plt.title(f'α={idx / (len(interpolated_results) - 1):.2f}')

    plt.suptitle('Latent Space Interpolation between Type 8 and Type 13')
    plt.show()

def plot_interpolated_hvf_arch(interpolated_samples, results_dir, file_name='interpolated_hvf_row_plot.png'):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Set up a figure for 10 plots in a row
    fig, axes = plt.subplots(1, len(interpolated_samples), figsize=(20, 5))  # N plots in a row

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mask = init_mask(device).cpu().numpy()

    for idx, sample in enumerate(interpolated_samples):
        # Convert the tensor to numpy if it is a torch.Tensor
        if isinstance(sample, torch.Tensor):
            hvf_image = sample.cpu().numpy()
        else:
            hvf_image = sample  # Already a numpy array

        # Apply the mask to the image
        hvf_image = hvf_image.squeeze()  # Remove unnecessary dimensions
        hvf_image_dis = np.where(mask == 1, hvf_image, np.nan)  # Apply the mask

        # Plotting each image
        ax = axes[idx]
        ax.imshow(hvf_image_dis, cmap='gray', interpolation='none', vmin=np.nanmin(hvf_image_dis), vmax=np.nanmax(hvf_image_dis))
        ax.set_title(f"α = {idx / (len(interpolated_samples) - 1):.2f}")  # Set the title as the interpolation value (alpha)
        ax.axis('off')  # Hide axes

    # Save the figure
    plt.savefig(os.path.join(results_dir, file_name))
    plt.close()

def unnormalize(image, mean, std):
    return image * std + mean

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