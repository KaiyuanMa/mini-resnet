import torch
import torch.nn as nn
import torch.optim as optim
from mini_resnet import mini_resnet


def make_dataset(n=600):
    X = torch.randn(n, 3, 32, 32)
    y = torch.randint(0, 10, (n,))
    return X, y


def train_model(model, X, y, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss={loss.item():.4f}")


if __name__ == "__main__":
    X, y = make_dataset()
    
    model = mini_resnet()
    print(model)
    
    train_model(model, X, y, epochs=8)
