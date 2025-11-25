import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from mini_resnet import mini_resnet

def get_cifar10_dataloaders(batch_size=64, subset_size=5000):
    # Image augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.49, 0.48, 0.44), (0.20, 0.19, 0.20)),
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49, 0.48, 0.44), (0.20, 0.19, 0.20)),
    ])

    train_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    val_full = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)
    
    # Since we use CPU to train, we only use a small part of the training data set
    if subset_size is not None and subset_size < len(train_full):
        train_indices = torch.randperm(len(train_full))[:subset_size].tolist()
        train = Subset(train_full, train_indices)
    else:
        train = train_full
    
    #  Use 10% of training data as validation
    val_indices = torch.randperm(len(val_full))[:subset_size//10].tolist()
    val = Subset(val_full, val_indices)
    
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
    
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader

def compute_accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

def train_model(model, train_loader, val_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    
    model = model.to('cpu')
    
    # Start training
    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        train_batches = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to('cpu'), targets.to('cpu')
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item()
            train_acc_sum += compute_accuracy(outputs.detach(), targets)
            train_batches += 1
        
        train_loss = train_loss_sum / train_batches
        train_acc = train_acc_sum / train_batches

        # Do validation
        model.eval()
        val_loss_sum = 0.0
        val_acc_sum = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to('cpu'), targets.to('cpu')
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss_sum += loss.item()
                val_acc_sum += compute_accuracy(outputs, targets)
                val_batches += 1
        
        val_loss = val_loss_sum / val_batches
        val_acc = val_acc_sum / val_batches

        print(f"Epoch {epoch+1:03d}/{epochs} | "
              f"train_loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.3f}")


if __name__ == "__main__":
    
    # Load the data
    print("Loading CIFAR-10 dataset...")
    train_loader, val_loader = get_cifar10_dataloaders(batch_size=64, subset_size=5000)
    
    # Create model
    model = mini_resnet()
    print(model)
    print(f"\nTraining on {len(train_loader.dataset)} samples, "
          f"validating on {len(val_loader.dataset)} samples\n")
    
    # Train
    train_model(model, train_loader, val_loader, epochs=20)
