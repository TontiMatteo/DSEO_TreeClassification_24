import torch
import torch.nn as nn
import torchvision.models as models
from FocalLoss import FocalLoss

class HierarchicalResNet(nn.Module):
    def __init__(self, num_bands=6, num_classes_level1=2, num_classes_level2=9, num_classes_level3=19, fine_tuning=False):
        super(HierarchicalResNet, self).__init__()
        
        # Load a pretrained ResNet (e.g., ResNet18)
        self.resnet = models.resnet18(pretrained=True)

        if not fine_tuning:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        # Modify input layer for custom number of bands
        self.input_conv = nn.Conv2d(
            in_channels=num_bands,  # Custom number of input channels
            out_channels=3,         # Match ResNet's expected input
            kernel_size=3, stride=1, padding=1
        )
        
        # Extract ResNet features
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Remove original FC layer

        # Define separate classifiers for each level
        self.classifier_level1 = nn.Linear(num_features, num_classes_level1)
        self.classifier_level2 = nn.Linear(num_features + num_classes_level1, num_classes_level2)
        self.classifier_level3 = nn.Linear(num_features + num_classes_level2, num_classes_level3)

    def forward(self, x):
        x = self.input_conv(x)
        features = self.resnet(x)  # Extract features

        # Predict Level 1 labels
        level1_out = self.classifier_level1(features)

        # Predict Level 2 labels using both features and Level 1 output
        level2_input = torch.cat((features, level1_out), dim=1)
        level2_out = self.classifier_level2(level2_input)

        # Predict Level 3 labels using both features and Level 2 output
        level3_input = torch.cat((features, level2_out), dim=1)
        level3_out = self.classifier_level3(level3_input)

        return level1_out, level2_out, level3_out


def model(dataloader, num_bands=6, num_classes_level1=2, num_classes_level2=9, num_classes_level3=19, fine_tuning=False):
    model = HierarchicalResNet(num_bands=num_bands,
                               num_classes_level1=num_classes_level1, 
                               num_classes_level2=num_classes_level2, 
                               num_classes_level3=num_classes_level3, 
                               fine_tuning=fine_tuning) 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = FocalLoss(gamma = 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        epoch_loss = 0.0
        for inputs, labels1, labels2, labels3 in dataloader:
            inputs, labels1, labels2, labels3 = inputs.to(device), labels1.to(device), labels2.to(device), labels3.to(device)
    
            optimizer.zero_grad()
            
            out1, out2, out3 = model(inputs)
    
            loss1 = criterion(out1, labels1)
            loss2 = criterion(out2, labels2)
            loss3 = criterion(out3, labels3)
            
            total_loss = loss1 + loss2 + loss3  # Sum the losses
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

    return model
        