import torch
from torch.utils.data import DataLoader, random_split
import random
import os
from config import init_mask
from vae_model import VAE
from train_utils import test_and_evaluate
from hvf_dataset import HVFDataset
from plot import plot_comparison
import numpy as np
import json

def calculate_mae(originals, reconstructions, mask, global_min, global_max):
    """Calculates Mean Absolute Error (MAE) for each original and reconstruction pair."""
    mae_list = []
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    if torch.is_tensor(global_min):
        global_min = global_min.cpu().numpy()
    if torch.is_tensor(global_max):
        global_max = global_max.cpu().numpy()

    for i in range(len(originals)):
        original = originals[i].squeeze()*mask
        print(f"ori{original}")
        reconstruction = reconstructions[i].squeeze()
        print(f"rec{reconstruction}")

        original_denorm = original * (global_max - global_min) + global_min
        reconstruction_denorm = reconstruction * (global_max - global_min) + global_min

        masked_original = original_denorm[mask != 0]
        masked_reconstruction = reconstruction_denorm[mask != 0]
        mae = np.mean(np.abs(masked_original - masked_reconstruction))
        mae_list.append(mae)
        print(f"Pair {i+1} MAE: {mae}")
    return mae_list

latent_dim = 10
batch_size = 100
model_save_path ='model_fold_4.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
static_mask = init_mask(device)

# Load the saved model
model = VAE(latent_dim=latent_dim).to(device)
model.load_state_dict(torch.load(model_save_path))
model.eval()


dataset_path = '../src/uwhvf/alldata.json'
full_dataset = HVFDataset(dataset_path)


num_test = int(0.1 * len(full_dataset))
num_train = len(full_dataset) - num_test

_, test_dataset = random_split(full_dataset, [num_train, num_test])


test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
test_details = test_and_evaluate(model, device, test_loader, static_mask)

sorted_details = sorted(test_details, key=lambda x: x['loss'])
best_5 = sorted_details[:5]
worst_5 = sorted_details[-5:]
random_5 = random.sample(sorted_details, 5)

# print(f"Best 5 losses: {[x['loss'] for x in best_5]}")
# print(f"Worst 5 losses: {[x['loss'] for x in worst_5]}")
# print(f"Random 5 losses: {[x['loss'] for x in random_5]}")

eval_results_dir = 'eval_visual'
best_5_originals = [entry['original'] for entry in best_5]
worst_5_originals = [entry['original'] for entry in worst_5]
random_5_originals = [entry['original'] for entry in random_5]

best_5_reconstructions = [entry['reconstruction'] for entry in best_5]
worst_5_reconstructions = [entry['reconstruction'] for entry in worst_5]
random_5_reconstructions = [entry['reconstruction'] for entry in random_5]

# Call plot_comparison for best, worst, and random sets
plot_comparison(best_5_originals, best_5_reconstructions, static_mask, "best_5", results_dir=eval_results_dir)
plot_comparison(worst_5_originals, worst_5_reconstructions, static_mask, "worst_5", results_dir=eval_results_dir)
plot_comparison(random_5_originals, random_5_reconstructions, static_mask, "random_5", results_dir=eval_results_dir)

mae_list = calculate_mae(best_5_originals, best_5_reconstructions, static_mask,full_dataset.global_min, full_dataset.global_max)
print(f"MAE list for best 5 pairs: {mae_list}")

# test_results_path = os.path.join(results_dir, 'test_results.json')
# with open(test_results_path, 'w') as f:
#     json.dump(test_details, f)
# print(f"Test details saved to {test_results_path}")
