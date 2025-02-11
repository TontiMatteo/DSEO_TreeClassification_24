import torch.nn as nn
import torch.nn.functional as F
import torch


# Define valid mappings for hierarchical constraints
valid_second_level = {
    0: [0, 1, 2, 3],  # Broadleaf → Beech, Oak, Long-lived Deciduous, Short-lived Deciduous
    1: [4, 5, 6, 7, 8] # Needleleaf → Fir, Larch, Spruce, Pine, Douglas Fir
}

valid_third_level = {
    0: [0],           # Beech → European Beech
    1: [1, 2, 3],     # Oak → Sessile Oak, English Oak, Red Oak
    2: [4, 5, 6, 7],  # Long-lived Deciduous → Sycamore Maple, European Ash, Linden, Cherry
    3: [8, 9, 10],    # Short-lived Deciduous → Alder, Poplar, Birch
    4: [11],         # Fir → Silver Fir
    5: [12, 13],     # Larch → European Larch, Japanese Larch
    6: [14],         # Spruce → Norway Spruce
    7: [15, 16, 17], # Pine → Scots Pine, Black Pine, Weymouth Pine
    8: [18]          # Douglas Fir → Douglas Fir
}

class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.conv = nn.Conv2d(18, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32, 2)  

    def forward(self, x):
        x = F.relu(self.conv(x))  
        x = x.mean(dim=[2,3])  
        x = self.fc(x)
        return x


class Model2(nn.Module):
    def __init__(self, num_classes_level2):
        super(Model2, self).__init__()
        self.conv = nn.Conv2d(18, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 + 2, 64)  
        self.fc2 = nn.Linear(64, num_classes_level2)

    def forward(self, x, model1_pred):
        x = F.relu(self.conv(x))
        x = x.mean(dim=[2,3])  
        x = torch.cat([x, model1_pred], dim=1)  
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)  

        # **Apply Hard Constraint**
        first_level_pred_idx = model1_pred.argmax(dim=1)  
        mask = torch.full_like(logits, float('-inf'))  
        for batch_idx, pred in enumerate(first_level_pred_idx):
            valid_classes = valid_second_level[pred.item()]  
            mask[batch_idx, valid_classes] = 0  
        
        logits = logits + mask  
        probs = F.softmax(logits, dim=1)  

        return probs


class Model3(nn.Module):
    def __init__(self, input_channels=18, num_classes=19, first_label_classes=2, second_label_classes=9):
        super(Model3, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 3 * 3 + first_label_classes + second_label_classes, 256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x, first_label_pred, second_label_pred):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.reshape(x.shape[0], -1)  

        x = torch.cat([x, first_label_pred, second_label_pred], dim=1)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        logits = self.fc3(x)

        # **Apply Hard Constraint**
        second_level_pred_idx = second_label_pred.argmax(dim=1)  
        mask = torch.full_like(logits, float('-inf'))  
        for batch_idx, pred in enumerate(second_level_pred_idx):
            valid_classes = valid_third_level[pred.item()]  
            mask[batch_idx, valid_classes] = 0  
        
        logits = logits + mask  
        probs = F.softmax(logits, dim=1)  

        return probs


def train_model1(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for epoch in range(num_epochs):
        total_loss, correct, total = 0.0, 0, 0
        
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)  
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(dataloader):.4f}, Acc = {correct/total:.4f}")

    return model


def train_model2(model1, model2, dataloader, criterion, optimizer, num_epochs=10):
    model1.eval()
    model2.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(num_epochs):
        total_loss, correct, total = 0.0, 0, 0
        
        for images, labels_level1, labels_level2 in dataloader:
            images = images.to(device)
            labels_level1, labels_level2 = labels_level1.to(device), labels_level2.to(device)

            with torch.no_grad():
                pred_level1 = model1(images)
                pred_level1_soft = torch.softmax(pred_level1, dim=1)

            optimizer.zero_grad()
            outputs = model2(images, pred_level1_soft)
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
    model1.eval()
    model2.eval()
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