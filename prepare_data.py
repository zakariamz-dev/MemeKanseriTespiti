import os
import shutil
import random

SOURCE_DIR = r"C:\Temp\Dataset_BUSI_with_GT"
TARGET_DIR = "./data"

# 1. Nettoyage complet du dossier cible
if os.path.exists(TARGET_DIR):
    shutil.rmtree(TARGET_DIR)
    print("Dossier ./data réinitialisé proprement.")

# Ajout de la classe 'normal'
classes = ['benign', 'malignant', 'normal']
split_ratio = 0.8  # 80% train, 20% test

# 2. Création des répertoires pour les 3 classes
for split in ['train', 'test']:
    for cls in classes:
        os.makedirs(os.path.join(TARGET_DIR, split, cls), exist_ok=True)

# 3. Filtrage et répartition des images pour les 3 classes
for cls in classes:
    cls_dir = os.path.join(SOURCE_DIR, cls)
    if not os.path.exists(cls_dir):
        print(f"Dossier introuvable : {cls_dir}")
        continue
    
    # Exclure les masques (_mask)
    all_files = os.listdir(cls_dir)
    images = [f for f in all_files if f.endswith('.png') and '_mask' not in f]
    
    random.seed(42)  # Résultat reproductible
    random.shuffle(images)
    split_idx = int(len(images) * split_ratio)
    
    train_imgs = images[:split_idx]
    test_imgs = images[split_idx:]
    
    for img in train_imgs:
        shutil.copy(os.path.join(cls_dir, img), os.path.join(TARGET_DIR, 'train', cls, img))
        
    for img in test_imgs:
        shutil.copy(os.path.join(cls_dir, img), os.path.join(TARGET_DIR, 'test', cls, img))

    print(f"Classe [{cls}] -> Train: {len(train_imgs)} images | Test: {len(test_imgs)} images")

print("\nPréparation multiclasse terminée avec succès !")