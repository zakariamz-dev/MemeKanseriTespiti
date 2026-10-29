# 🩺 Système de Diagnostic Précoce du Cancer du Sein (Edge AI)

Système intelligent et embarquable de classification d'imagerie médicale (échographies et mammographies) basé sur une architecture **ResNet-18** avec **Transfer Learning**, optimisé pour l'inférence ultra-rapide sur la périphérie (**Edge AI** via ONNX Runtime).
---

## 🏗️ Architecture du Projet

Le projet adopte une architecture logicielle centralisée et épurée de niveau ingénieur :
```text
├── checkpoints/              # Poids PyTorch (.pth) et modèle optimisé ONNX (.onnx)
├── data/                     # Jeux de données d'entraînement et de test
│   ├── train/                # Dossier des images d'entraînement (benign, malignant)
│   └── test/                 # Dossier des images de test
├── venv/                     # Environnement virtuel Python
├── main.py                   # Script maître unique (CLI) pour tout le pipeline
├── app.py                    # Interface web interactive (Streamlit)
├── gesture_detection.py      # Scanner vidéo interactif en temps réel (OpenCV / PyTorch)
├── requirements.txt          # Dépendances du projet
└── README.md                 # Documentation technique