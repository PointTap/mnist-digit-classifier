import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load MNIST Dataset with Augmentation
# -----------------------------
transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1,0.1)),
    transforms.ToTensor()
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# -----------------------------
# 2. Define Improved Classification Autoencoder
# -----------------------------
class ClassifierAutoencoder(nn.Module):
    def __init__(self):
        super(ClassifierAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1, output_padding=0), nn.Sigmoid()
        )
        # Classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        classification = self.classifier(latent)
        return reconstruction[:, :, :28, :28], classification

# -----------------------------
# 3. Training Setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ClassifierAutoencoder().to(device)

recon_loss_fn = nn.BCELoss()
class_loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# -----------------------------
# 4. Training Loop
# -----------------------------
epochs = 30
train_losses = []
train_accs = []

for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    model.train()
    for data, targets in train_loader:
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        recon, preds = model(data)
        loss_recon = recon_loss_fn(recon, data)
        loss_class = class_loss_fn(preds, targets)
        # Weight classification higher than reconstruction
        loss = 0.3 * loss_recon + 0.7 * loss_class
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = preds.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    scheduler.step()
    avg_loss = running_loss / len(train_loader)
    acc = 100. * correct / total
    train_losses.append(avg_loss)
    train_accs.append(acc)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%")

# -----------------------------
# 5. Plot Loss & Accuracy
# -----------------------------
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(range(1, epochs+1), train_losses)
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training Loss")
plt.subplot(1,2,2)
plt.plot(range(1, epochs+1), train_accs)
plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.title("Training Accuracy")
plt.show()

# -----------------------------
# 6. Test Evaluation
# -----------------------------
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, targets in test_loader:
        data, targets = data.to(device), targets.to(device)
        _, preds = model(data)
        _, predicted = preds.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
print(f"Test Accuracy: {100.*correct/total:.2f}%")

# -----------------------------
# 7. Save the model
# -----------------------------
torch.save(model.state_dict(), "mnist_autoencoder_classifier.pth")
