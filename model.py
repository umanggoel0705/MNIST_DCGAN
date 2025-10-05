import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(1, 64, 5, 2, 2), # 28*28*1 -> 13*13*64
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(64, 128, 5, 2, 2), # 13*13*64 -> 7*7*128
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(128*7*7, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)
    
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(100, 128*7*7),
            nn.LeakyReLU(),
            nn.Unflatten(1, (128,7,7)),
            nn.ConvTranspose2d(128, 64, 5, 1, 2, 0),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 5, 2, 2, 1),
            nn.BatchNorm2d(32, momentum=0.1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 1, 5, 2, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)
    
# Hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 64
noise_dim = 100
lr = 2e-4
num_epochs = 50

# Initialization
disc = Discriminator().to(device)
gen = Generator().to(device)
fixed_noise = torch.randn((batch_size, noise_dim)).to(device)

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
)
dataset = datasets.MNIST(root='datasets/', transform=transform, download=True)
loader = DataLoader(dataset, batch_size, shuffle=True)

opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()

writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake") # store fake generated images
writer_real = SummaryWriter(f"runs/GAN_MNIST/real") # store real images
step = 0

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        batch_size = real.shape[0]

        # Train Discriminator
        noise = torch.randn((batch_size, noise_dim)).to(device)
        fake = gen(noise)

        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_fake + lossD_real) / 2

        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()

        # Train Generator
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        if batch_idx == 0:
            print(
                f"Epoch: [{epoch}/{num_epochs}] \n "
                f"Loss D:{lossD:.4f} Loss G:{lossG:.4f}"
                )
            
            fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
            data = real.reshape(-1, 1, 28, 28)

            img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
            img_grid_real = torchvision.utils.make_grid(data, normalize=True)

            writer_fake.add_image(
                "MNIST fake img", img_grid_fake, global_step=step
            )

            writer_real.add_image(
                "MNIST real img", img_grid_real, global_step=step
            )

            step += 1