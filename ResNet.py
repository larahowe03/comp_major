import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit

# 1. Config
data_dir = 'chess_piece_images'
test_size = 0.2  # 20% validation
batch_size = 16
epochs = 10
lr = 0.001
seed = 42
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 2. Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(15),             # small random rotations
    transforms.ColorJitter(brightness=0.2,
                           contrast=0.2,
                           saturation=0.2),   # lighting variation
    transforms.RandomHorizontalFlip(),         # mirror variations
    transforms.RandomVerticalFlip(),           # optional (if top-down view)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 3. Load everything
full_dataset = datasets.ImageFolder(data_dir, transform=transform)
targets = [label for _, label in full_dataset.samples]

# 4. Stratified split
splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
train_idx, val_idx = next(splitter.split(torch.zeros(len(targets)), targets))

train_dataset = Subset(full_dataset, train_idx)
val_dataset   = Subset(full_dataset, val_idx)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)}")
print("Detected classes:", full_dataset.classes)

# ----------------------------
# 3. Model setup (ResNet18)
# ----------------------------
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False   # freeze backbone

num_classes = len(full_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=lr)

# ----------------------------
# 4. Training loop
# ----------------------------
for epoch in range(epochs):
    model.train()
    running_loss, correct, total = 0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_acc = 100. * correct / total
    print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.3f}  Acc: {train_acc:.2f}%")

# ----------------------------
# 5. Save model
# ----------------------------
torch.save(model.state_dict(), 'resnet_chess.pkl')
print("Training complete and model saved as resnet_chess.pkl")