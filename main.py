import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Transformations adaptées à ResNet-18 (224x224, 3 canaux, normalisation ImageNet)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # ResNet attend 3 canaux RVB
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. Chargement des données (3 classes : benign, malignant, normal)
train_dataset = datasets.ImageFolder(root='./data/train', transform=transform)
test_dataset = datasets.ImageFolder(root='./data/test', transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)

# 3. Chargement de ResNet-18 pré-entraîné
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Modification de la dernière couche fully connected pour nos 3 classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)

# Poids pour rééquilibrer les 3 classes [benign, malignant, normal]
class_weights = torch.tensor([1.0, 1.5, 3.5])
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Optimiseur avec un learning rate plus doux pour le fine-tuning
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 4. Entraînement du modèle (15 à 20 époques suffisent généralement avec le Transfer Learning)
num_epochs = 20
print(f"--- Début du Transfer Learning avec ResNet-18 ({num_epochs} époques) ---")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    print(f"Époque [{epoch+1}/{num_epochs}], Perte moyenne: {running_loss/len(train_loader):.4f}")

print("--- Fin de l'entraînement ---\n")

# 5. Évaluation détaillée et génération de la Matrice de Confusion
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("--- Évaluation médicale détaillée (ResNet-18) ---")
print("Classes :", train_dataset.classes)

cm = confusion_matrix(all_labels, all_preds)

print("\nRapport de Classification :")
print(classification_report(all_labels, all_preds, target_names=train_dataset.classes, zero_division=0))

# 6. Affichage graphique de la matrice de confusion
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=train_dataset.classes, 
            yticklabels=train_dataset.classes)
plt.title('Matrice de Confusion - ResNet-18 (Transfer Learning)')
plt.xlabel('Prédictions du Modèle')
plt.ylabel('Vérités Terrain (Réalité)')
plt.tight_layout()
plt.show()
torch.save(model.state_dict(), "checkpoints/model_resnet18.pth")