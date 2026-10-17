import torch
import torch.nn as nn

class MedicalCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(MedicalCNN, self).__init__()
        
        # 1 canal d'entrée (pour les images en niveaux de gris / Grayscale)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Couche de Dropout pour éviter le surapprentissage (très important en imagerie médicale)
        self.dropout = nn.Dropout(0.25)
        
        # Note : Si vos images font 128x128 pixels en entrée, 
        # après deux max-pooling de 2x2, la taille devient 32 * 32 * 32.
        # Ajustez la dimension linéaire en fonction de la taille de vos images d'entrée.
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes) # 2 classes de sortie (ex: Bénin / Malin)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        
        # Aplatissement dynamique pour s'adapter au batch
        x = x.view(x.size(0), -1)
        
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x