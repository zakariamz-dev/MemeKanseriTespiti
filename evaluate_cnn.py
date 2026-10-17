import torch
import torch.nn as nn
import torch.optim as optim

# 1. Définir une architecture de réseau de neurones simple
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return out

# 2. Paramètres du modèle
input_dim = 10    # Nombre de caractéristiques en entrée
hidden_dim = 16   # Nombre de neurones cachés
output_dim = 2    # Nombre de classes en sortie
learning_rate = 0.01

# Instanciation du modèle
model = SimpleNN(input_dim, hidden_dim, output_dim)
print("Modèle initialisé avec succès :")
print(model)

# 3. Définir la fonction de perte et l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 4. Simulation de données d'entraînement (100 échantillons, 10 caractéristiques)
X_train = torch.randn(100, input_dim)
y_train = torch.randint(0, output_dim, (100,))

# 5. Boucle d'entraînement simple (ex: 5 époques)
epochs = 5
print("\nDébut de l'entraînement...")

for epoch in range(epochs):
    # Passage avant (Forward pass) : calcul des prédictions
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Passage arrière (Backward pass) et optimisation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Époque [{epoch+1}/{epochs}], Perte (Loss): {loss.item():.4f}")

print("\nEntraînement terminé avec succès !")