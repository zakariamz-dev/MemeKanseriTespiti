# 🩺 Détection Précoce du Cancer du Sein par Échographie (Deep Learning, Edge AI & XAI)

Ce projet s'inscrit dans le cadre d'un travail de recherche de Master en Génie Informatique. Il implémente un système d'aide au diagnostic médical basé sur l'apprentissage profond (**Deep Learning**) pour la classification d'images d'échographie mammaire (classes : *Benign*, *Malignant*, *Normal*). L'approche met un accent particulier sur l'embarquabilité sur dispositifs portables (**Edge AI / TinyML**), l'optimisation méta-heuristique (**PSO**) et l'explicabilité médicale (**XAI / Grad-CAM**).

---

## 📂 Architecture du Projet

```text
Meme Kanseri Tespiti/
├── checkpoints/
│   ├── model_resnet18.pth             # Modèle de base entraîné (FP32)
│   └── model_resnet18_quantized.pth   # Modèle optimisé et quantifié (INT8)
├── data/                              # Jeux de données (train / test structurés par classes)
│   ├── train/                         # benign, malignant, normal
│   └── test/                          # benign, malignant, normal
├── venv/                              # Environnement virtuel Python
├── .gitignore                         # Fichiers ignorés par Git
├── evaluate_cnn.py                    # Script d'évaluation et métriques de performance (Matrice de confusion)
├── grad_cam.py                        # Module d'explicabilité XAI (Génération automatique des cartes Grad-CAM)
├── main.py                            # Pipeline principal d'entraînement et d'évaluation
├── models.py                          # Définition des architectures de réseaux de neurones
├── optimize_edge.py                   # Script d'optimisation Edge AI (Quantification PTQ)
├── prepare_data.py                    # Prétraitement et normalisation des échographies
├── pso_optimizer.py                   # Optimisation par essaim de particules (PSO) pour les hyperparamètres
├── README.md                          # Documentation du projet
└── requirements.txt                   # Dépendances et bibliothèques du projet