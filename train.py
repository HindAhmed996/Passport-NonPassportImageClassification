import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from tqdm import tqdm
from PIL import ImageFile
from sklearn.metrics import confusion_matrix, precision_score, recall_score

ImageFile.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the data
train_data = datasets.ImageFolder('../train', transform=transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]))
train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)
print('loaded training data')
test_data = datasets.ImageFolder('../test', transform=transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]))
test_loader = torch.utils.data.DataLoader(test_data, batch_size=8, shuffle=False)

# Define the model
model = models.resnet101(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model.to(device)

# Define the loss function and optimizer

# criterion = nn.BCELoss()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)

# Train the model
num_epochs = 20
for epoch in tqdm(range(num_epochs)):
    running_loss = 0.0
    model.train() #........................................
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        # loss = criterion(torch.argmax(outputs, dim=1), labels.float())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print statistics every epoch
    print(f"Epoch {epoch+1} loss: {running_loss/len(train_loader)}")

    # Evaluate the model
    correct = 0
    total = 0
    predicted_labels = []
    ground_truth_labels = []
    
    with torch.no_grad():
        model.eval() #........................................
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            predicted_labels.extend(predicted.tolist())
            ground_truth_labels.extend(labels.tolist())

    # Calculate metrics
    cm = confusion_matrix(ground_truth_labels, predicted_labels)
    precision = precision_score(ground_truth_labels, predicted_labels)
    recall = recall_score(ground_truth_labels, predicted_labels)
    torch.save(model.state_dict(), f'passport_classifier_{epoch}.pth')
    print(f"Accuracy on test set: {100 * correct / total}%")
    print(f"Confusion matrix:\n{cm}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")