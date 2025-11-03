import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

# ----------------------------
# 1️⃣ Configuration
# ----------------------------
data_dir = 'big_chess_piece_dataset_png'
batch_size = 16
lr = 0.001
seed = 42
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(seed)

# ----------------------------
# 2️⃣ Data transforms
# ----------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ----------------------------
# 3️⃣ Dataset + Stratified Split (70/20/10)
# ----------------------------
full_dataset = datasets.ImageFolder(data_dir, transform=train_transform)
targets = [label for _, label in full_dataset.samples]

# Step 1: Split off test set (10%)
trainval_idx, test_idx = train_test_split(
    range(len(targets)), test_size=0.1, stratify=targets, random_state=seed
)

# Step 2: Split remaining 90% into train (70% of total) and val (20% of total)
# 20/90 ≈ 0.222 to get 20% of original
train_idx, val_idx = train_test_split(
    trainval_idx, test_size=0.222, stratify=[targets[i] for i in trainval_idx], random_state=seed
)

# Build subsets
train_dataset = Subset(full_dataset, train_idx)
val_dataset   = Subset(datasets.ImageFolder(data_dir, transform=val_transform), val_idx)
test_dataset  = Subset(datasets.ImageFolder(data_dir, transform=val_transform), test_idx)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Dataset split: Train={len(train_dataset)} ({len(train_dataset)/len(targets)*100:.1f}%) | "
      f"Val={len(val_dataset)} ({len(val_dataset)/len(targets)*100:.1f}%) | "
      f"Test={len(test_dataset)} ({len(test_dataset)/len(targets)*100:.1f}%)")


# ----------------------------
# 4️⃣ Model setup
# ----------------------------
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Freeze backbone for first stage
for param in model.parameters():
    param.requires_grad = False

num_classes = len(full_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)


class EarlyStopper:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state_dict = None
        self.best_epoch = 0

    def __call__(self, val_acc, model, epoch):
        if self.best_score is None:
            self.best_score = val_acc
            self.best_state_dict = model.state_dict()
            self.best_epoch = epoch
        elif val_acc < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_acc
            self.best_state_dict = model.state_dict()
            self.best_epoch = epoch
            self.counter = 0

# ----------------------------
# 5️⃣ Evaluation function
# ----------------------------
def evaluate(model, loader):
    model.eval()
    correct, total, val_loss = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return val_loss / len(loader), 100. * correct / total

# ----------------------------
# 6️⃣ Stage 1: Train FC layer only
# ----------------------------
print("\n" + "="*70)
print("[Stage 1] Training FC layer only (backbone frozen)")
print("="*70)
print(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12}")
print("-" * 70)

early_stopper = EarlyStopper(patience=5, min_delta=0.1)

epoch = 0
while not early_stopper.early_stop:
    epoch += 1
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

    scheduler.step()
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    val_loss, val_acc = evaluate(model, val_loader)

    print(f"{epoch:<8} {train_loss:<12.4f} {train_acc:<12.2f}% {val_loss:<12.4f} {val_acc:<12.2f}%")

    early_stopper(val_acc, model, epoch)

print("-" * 70)
print(f"Early stopping triggered at epoch {epoch}")
print(f"Best model: Epoch {early_stopper.best_epoch} with Val Acc = {early_stopper.best_score:.2f}%")
    
if early_stopper.best_state_dict is not None:
    model.load_state_dict(early_stopper.best_state_dict)
    print(f"Restored best model from epoch {early_stopper.best_epoch}")

# ----------------------------
# 7️⃣ Stage 2: Fine-tune full network
# ----------------------------
print("\n" + "="*70)
print("[Stage 2] Fine-tuning full ResNet (all layers unfrozen)")
print("="*70)
print(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12}")
print("-" * 70)

early_stopper_ft = EarlyStopper(patience=5, min_delta=0.1)

for param in model.parameters():
    param.requires_grad = True

optimizer = optim.Adam(model.parameters(), lr=lr * 0.1)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

epoch = 0
while not early_stopper_ft.early_stop:
    epoch += 1
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
    
    scheduler.step()
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    val_loss, val_acc = evaluate(model, val_loader)
    
    print(f"{epoch:<8} {train_loss:<12.4f} {train_acc:<12.2f}% {val_loss:<12.4f} {val_acc:<12.2f}%")
    
    early_stopper_ft(val_acc, model, epoch)

print("-" * 70)
print(f"Early stopping triggered at epoch {epoch}")
print(f"Best model: Epoch {early_stopper_ft.best_epoch} with Val Acc = {early_stopper_ft.best_score:.2f}%")

# Restore best fine-tuned model
if early_stopper_ft.best_state_dict is not None:
    model.load_state_dict(early_stopper_ft.best_state_dict)
    print(f"Restored best fine-tuned model from epoch {early_stopper_ft.best_epoch}")

# ----------------------------
# 8️⃣ Final Test Evaluation
# ----------------------------
print("\n" + "="*70)
print("Final Evaluation on Test Set")
print("="*70)

test_loss, test_acc = evaluate(model, test_loader)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.2f}%")

# ----------------------------
# 9️⃣ Save ONLY the best model
# ----------------------------
model_path = 'resnet_chess_best.pth'
torch.save(model.state_dict(), model_path)
print(f"\n✅ Best model saved as '{model_path}'")
print(f"   Best epoch from Stage 2: {early_stopper_ft.best_epoch}")
print(f"   Best validation accuracy: {early_stopper_ft.best_score:.2f}%")
print(f"   Final test accuracy: {test_acc:.2f}%")