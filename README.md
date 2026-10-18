# 🩺 Détection Précoce du Cancer du Sein par Vision par Ordinateur et IA Embarquée (Edge AI)

## 📌 À propos du projet
Ce projet s'inscrit dans le cadre d'un mémoire de Master en Génie Informatique. Il propose un système d'aide au diagnostic pour la **détection précoce du cancer du sein** à partir d'images d'échographie médicale. L'application combine un modèle de Deep Learning basé sur **ResNet-18** avec une interface interactive en temps réel pilotée par reconnaissance de gestes et vision par ordinateur (`OpenCV`), conçue pour être déployée sur des dispositifs portables (Edge AI).

---

## 🛠️ Architecture du Projet

```text
Meme Kanseri Tespiti/
│
├── data/                    # Données d'entraînement, de validation et de test (Bénin, Malin, Normal)
├── .gitignore               # Fichiers et dossiers exclus du versionnement Git (cache, temporaires)
├── README.md                # Documentation du projet
├── evaluate_cnn.py          # Script d'évaluation des performances du modèle
├── gesture_detection.py     # Script principal (Interface interactive, détection de mouvement et inférence IA)
├── main.py                  # Script d'entraînement et d'optimisation du réseau de neurones
├── model_resnet18.pth       # Poids entraînés du modèle ResNet-18
├── models.py                # Définition de l'architecture du réseau de neurones
└── prepare_data.py          # Script de prétraitement et de normalisation des images médicales