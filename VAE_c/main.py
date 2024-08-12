# Setup command line arguments
parser = argparse.ArgumentParser(description='VAE HVF Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--no-mps', action='store_true', default=False,
                    help='disables macOS GPU training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
use_mps = not args.no_mps and torch.backends.mps.is_available()

torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "mps" if use_mps else "cpu")

# Load the full dataset
full_dataset = HVFDataset('../src/uwhvf/alldata.json')

# Define the size of your test set
num_test = int(0.2 * len(full_dataset))  # Let's say 20% of the data
num_train = len(full_dataset) - num_test

# Split the dataset
train_dataset, test_dataset = random_split(full_dataset, [num_train, num_test])

# Setup DataLoaders
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Initialize the model and optimizer
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Main training and testing loop
if __name__ == "__main__":
    # for epoch in range(1, args.epochs + 1):
    #     train(model, device, train_loader, optimizer, epoch, args.log_interval)
    #     test(model, device, test_loader, epoch)
    originals = None
    reconstructions = {i: [] for i in range(10)}  # Assuming 10 test cases as an example
    results_dir = 'results_vae_11*10'
    os.makedirs(results_dir, exist_ok=True)  # Ensure the results directory is created
    train_losses = []
    latent_mean = 0
    latent_logvar = 0
    latent_std = 0
    total_data_count = 0

    # for epoch in range(1, args.epochs + 1):
    #     train_loss = train(model, device, train_loader, optimizer, epoch, args.log_interval)
    #     originals = test(model, device, test_loader, epoch, reconstructions, originals, results_dir)
    #     train_losses.append(train_loss)

    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch, args.log_interval)
        originals = test(model, device, test_loader, epoch, reconstructions, originals, results_dir)
        train_losses.append(train_loss)

        # Reset the sum accumulators and the count at the start of each epoch
        latent_mean = 0
        latent_logvar = 0
        total_data_count = 0

        for _, inputs in enumerate(train_loader):
            inputs = inputs.to(device)
            _, mu, logvar = model(inputs)
            latent_mean += mu.sum(0)
            latent_logvar += logvar.sum(0)
            total_data_count += inputs.size(0)

        # Calculating the mean and standard deviation after processing all batches
        if epoch == args.epochs:  # Only at the last epoch
            latent_mean /= total_data_count
            latent_std = torch.exp(0.5 * latent_logvar / total_data_count)

    if originals is not None:
        plot_all_reconstructions(originals, reconstructions, args.epochs, results_dir)
    else:
        print("Error: Original data not captured correctly.")

    num_samples = 10
    # sampled_data = sample_from_latent_space(model, device, num_samples=num_samples)
    # sampled_data = sample_from_latent_space_adjusted(model, device, num_samples=num_samples,latent_mean, latent_std)
    # print(latent_mean)
    # print(latent_std)
    sampled_data = sample_from_latent_space_adjusted(
        model=model,
        device=device,
        num_samples=num_samples,
        latent_mean=latent_mean,
        latent_std=latent_std
    )

    plot_samples(sampled_data)
    print("Sampled Data Shape:", sampled_data.shape)
    visualize_latent_space(model, test_loader, device)
    save_loss_plot(train_losses, 'Loss/train')