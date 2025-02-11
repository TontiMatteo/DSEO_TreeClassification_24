import torch.nn as nn
import torch.nn.functional as F
import torch



class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.conv = nn.Conv2d(18, 32, kernel_size=3, padding=1)  # 18-band input
        self.fc = nn.Linear(32, 2)  # Binary classification output

    def forward(self, x):
        x = F.relu(self.conv(x))  # Feature extraction
        x = x.mean(dim=[2,3])  # Global average pooling
        x = self.fc(x)
        return x

class Model2(nn.Module):
    def __init__(self, num_classes_level2):
        super(Model2, self).__init__()
        self.conv = nn.Conv2d(18, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 + 2, 64)  # Concatenate Model 1's prediction
        self.fc2 = nn.Linear(64, num_classes_level2)

    def forward(self, x, model1_pred):
        x = F.relu(self.conv(x))
        x = x.mean(dim=[2,3])  # Global average pooling
        x = torch.cat([x, model1_pred], dim=1)  # Concatenate Model 1 output
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Model3(nn.Module):
    def __init__(self, input_channels=18, num_classes=19, first_label_classes=2, second_label_classes=9):
        """
        Third hierarchical model that predicts the third-level label based on:
        - The original image (input_channels = 18)
        - The first-level prediction (first_label_classes)
        - The second-level prediction (second_label_classes)
        """
        super(Model3, self).__init__()

        # Feature extraction - Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Additional conv layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3 + first_label_classes + second_label_classes, 256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)  # Output layer

    def forward(self, x, first_label_pred, second_label_pred):
        """
        Forward pass.
        x: Image tensor (batch, 18, 3, 3)
        first_label_pred: One-hot encoding of first-level prediction
        second_label_pred: One-hot encoding of second-level prediction
        """
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Flatten image features
        x = x.reshape(x.shape[0], -1)  # (batch, 128 * 3 * 3)

        # Concatenate with hierarchical predictions
        x = torch.cat([x, first_label_pred, second_label_pred], dim=1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)  # No softmax (handled by loss function)

        return x


def train_model1(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for epoch in range(num_epochs):
        total_loss, correct, total = 0.0, 0, 0
        
        for images, labels in dataloader:  # Images: (B, 18, 3, 3), Labels: (B,)
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)  # Output logits
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Compute accuracy
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(dataloader):.4f}, Acc = {correct/total:.4f}")

    return model

    
def train_model2(model1, model2, dataloader, criterion, optimizer, num_epochs=10):
    model1.eval()  # Freeze Model 1
    model2.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(num_epochs):
        total_loss, correct, total = 0.0, 0, 0
        
        for images, labels_level1, labels_level2 in dataloader:
            images = images.to(device)
            labels_level1, labels_level2 = labels_level1.to(device), labels_level2.to(device)

            with torch.no_grad():
                pred_level1 = model1(images)  # Get Model 1's prediction
                pred_level1_soft = torch.softmax(pred_level1, dim=1)  # Soft input

            optimizer.zero_grad()
            outputs = model2(images, pred_level1_soft)  # Use Model 1 output
            loss = criterion(outputs, labels_level2)
            
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels_level2).sum().item()
            total += labels_level2.size(0)
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(dataloader):.4f}, Acc = {correct/total:.4f}")

    return model2

def train_model3(model1, model2, model3, dataloader, criterion, optimizer, num_epochs=10):
    model1.eval()  # Freeze Model 1
    model2.eval()  # Freeze Model 2
    model3.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(num_epochs):
        total_loss, correct, total = 0.0, 0, 0
        
        for images, labels_level1, labels_level2, labels_level3 in dataloader:
            images = images.to(device)
            labels_level1 = labels_level1.to(device)
            labels_level2 = labels_level2.to(device)
            labels_level3 = labels_level3.to(device)

            with torch.no_grad():
                pred_level1 = torch.softmax(model1(images), dim=1)
                pred_level2 = torch.softmax(model2(images, pred_level1), dim=1)

            optimizer.zero_grad()
            outputs = model3(images, pred_level1, pred_level2)
            loss = criterion(outputs, labels_level3)
            
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels_level3).sum().item()
            total += labels_level3.size(0)
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(dataloader):.4f}, Acc = {correct/total:.4f}")

    return model3