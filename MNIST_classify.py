import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load MNIST Dataset
# -----------------------------
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# -----------------------------
# 2. Define Classification Autoencoder
# -----------------------------
class ClassifierAutoencoder(nn.Module):
    def __init__(self):
        super(ClassifierAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1, output_padding=0), nn.Sigmoid()
        )
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 128), nn.ReLU(),
            nn.Linear(128, 10)  # 10 digits
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
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# -----------------------------
# 4. Training Loop
# -----------------------------
epochs = 20
train_losses = []
train_accs = []

for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    for data, targets in train_loader:
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        recon, preds = model(data)
        loss_recon = recon_loss_fn(recon, data)
        loss_class = class_loss_fn(preds, targets)
        loss = loss_recon + loss_class
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = preds.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
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