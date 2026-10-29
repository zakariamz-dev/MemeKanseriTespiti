import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. ENTRAÎNEMENT ---
def train(epochs=10):
    print("="*60)
    print("🚀 PIPELINE : ENTRAÎNEMENT DU MODÈLE RESNET-18")
    print("="*60)
    
    device = get_device()
    print(f"[INFO] Utilisation du périphérique : {device}")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dir = "data/train"
    if not os.path.exists(train_dir):
        print(f"[ERREUR] Dossier '{train_dir}' introuvable.")
        return

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    print(f"[INFO] Images d'entraînement : {len(train_dataset)} | Classes : {train_dataset.classes}")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    os.makedirs("checkpoints", exist_ok=True)
    weights_path = "checkpoints/resnet18_best.pth"
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            total += inputs.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct.double() / total
        scheduler.step()

        print(f"Époque [{epoch+1}/{epochs}] | Loss: {epoch_loss:.4f} | Acc: {epoch_acc * 100:.2f}%")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), weights_path)
            print(f"  --> Meilleur modèle sauvegardé : {weights_path}")

    print("="*60)
    print("✨ Entraînement terminé avec succès !")

# --- 2. ÉVALUATION ---
def evaluate():
    print("="*60)
    print("📊 PIPELINE : ÉVALUATION ET MATRICE DE CONFUSION")
    print("="*60)
    
    device = get_device()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dir = "data/test"
    if not os.path.exists(test_dir):
        print(f"[ERREUR] Dossier '{test_dir}' introuvable.")
        return

    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(test_dataset.classes))
    
    weights_path = "checkpoints/resnet18_best.pth"
    if not os.path.exists(weights_path):
        print(f"[ERREUR] Poids introuvables à {weights_path}. Entraînez d'abord le modèle.")
        return

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = np.sum(np.array(all_preds) == np.array(all_labels)) / len(test_dataset)
    print(f"Exactitude Globale (Accuracy) : {accuracy * 100:.2f}%")
    print("-" * 60)
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

    # Affichage graphique de la matrice de confusion
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_dataset.classes)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
    plt.title("Matrice de Confusion - ResNet-18 (Edge AI)")
    plt.tight_layout()
    plt.show()

# --- 3. OPTIMISATION EDGE (ONNX) ---
def optimize():
    print("="*60)
    print("⚙️ PIPELINE : EXPORTATION ET OPTIMISATION EDGE (ONNX)")
    print("="*60)

    device = get_device()
    weights_path = "checkpoints/resnet18_best.pth"
    if not os.path.exists(weights_path):
        print(f"[ERREUR] Modèle introuvable à {weights_path}.")
        return

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    os.makedirs("checkpoints", exist_ok=True)
    onnx_path = "checkpoints/model_resNet18_optimized.onnx"
    
    torch.onnx.export(
        model, dummy_input, onnx_path,
        export_params=True, opset_version=11,
        do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    original_size = os.path.getsize(weights_path) / (1024 * 1024)
    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
    
    print(f"[INFO] Modèle PyTorch d'origine : {original_size:.2f} MB")
    print(f"[INFO] Modèle ONNX optimisé Edge : {onnx_size:.2f} MB")
    print(f"✨ Exportation réussie vers : {onnx_path}")

# --- 4. TEST D'INFÉRENCE ONNX ---
def test_onnx():
    import onnxruntime as ort  # Importation paresseuse sécurisée
    print("="*60)
    print("🧪 PIPELINE : TEST D'INFÉRENCE ONNX")
    print("="*60)

    onnx_path = "checkpoints/model_resNet18_optimized.onnx"
    if not os.path.exists(onnx_path):
        print(f"[ERREUR] Modèle ONNX introuvable. Lancez d'abord l'optimisation (--mode optimize).")
        return

    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(root="data/test", transform=transform)
    classes = test_dataset.classes

    for i in range(min(5, len(test_dataset))):
        img_tensor, true_label = test_dataset[i]
        img_np = img_tensor.unsqueeze(0).numpy()
        
        outputs = ort_session.run(None, {input_name: img_np})
        pred_idx = np.argmax(outputs[0], axis=1)[0]
        
        status = "✅" if pred_idx == true_label else "❌"
        print(f"[{i+1}] Réalité: {classes[true_label]:<10} | Prédiction ONNX: {classes[pred_idx]:<10} | {status}")

# --- POINT D'ENTRÉE CLI ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Master - ResNet-18 Breast Cancer Detection")
    parser.add_argument("--mode", type=str, required=True, 
                        choices=["train", "evaluate", "optimize", "test"],
                        help="Mode d'exécution du pipeline")
    parser.add_argument("--epochs", type=int, default=10, help="Nombre d'époques pour l'entraînement")
    
    args = parser.parse_args()

    if args.mode == "train":
        train(epochs=args.epochs)
    elif args.mode == "evaluate":
        evaluate()
    elif args.mode == "optimize":
        optimize()
    elif args.mode == "test":
        test_onnx()