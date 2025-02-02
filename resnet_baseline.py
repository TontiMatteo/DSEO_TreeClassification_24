from torch import nn
from torchvision import models
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch
from FocalLoss import FocalLoss

class CustomResNet(nn.Module):
    def __init__(self, num_bands=6, num_classes=19, ft = False):
        super(CustomResNet, self).__init__()
        
        # Load a pretrained ResNet (e.g., ResNet18)
        self.resnet = models.resnet18(pretrained=True)
        
        # Freeze all the pretrained ResNet weights
        if not ft:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        # Add a new convolutional layer at the beginning
        self.input_conv = nn.Conv2d(
            in_channels=num_bands,  # Input bands (6 in your case)
            out_channels=3,        # Output channels to match ResNet input
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        # Replace the final fully connected layer
        num_features = self.resnet.fc.in_features  # Number of features in ResNet's fc layer
        self.resnet.fc = nn.Linear(num_features, num_classes)  # Output 19 classes
        
    def forward(self, x):
        # Pass through the custom input layer
        x = self.input_conv(x)
        x = self.resnet(x)  # Pass through the ResNet
        return x

def model(dataloader, num_bands=6, num_classes=19, fine_tuning=False):
    
    # Initialize the model
    model = CustomResNet(num_bands=num_bands, num_classes=num_classes, ft=fine_tuning)
    
    # Define device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Define the loss function and optimizer
    criterion = FocalLoss(gamma = 2)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),  # Trainable parameters only
        lr=1e-3
    )
    
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        epoch_loss = 0.0
        
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate loss for the epoch
            epoch_loss += loss.item()
        
        # Print epoch loss
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.4f}")
    
    return model
    