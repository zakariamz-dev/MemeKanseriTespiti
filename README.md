# 🩺 Détection Précoce du Cancer du Sein par Échographie (Deep Learning & Edge AI)

Ce projet s'inscrit dans le cadre d'un travail de recherche de Master en Génie Informatique. Il implémente un système d'aide au diagnostic médical basé sur l'apprentissage profond (**Deep Learning**) pour la classification d'images d'échographie mammaire (classes : *Benign*, *Malignant*, *Normal*). L'approche met un accent particulier sur l'embarquabilité sur dispositifs portables (**Edge AI / TinyML**) et l'optimisation méta-heuristique.

---

## 📂 Architecture du Projet

```text
Meme Kanseri Tespiti/
├── checkpoints/
│   ├── model_resnet18.pth             # Modèle de base entraîné (FP32)
│   └── model_resnet18_quantized.pth   # Modèle optimisé et quantifié (INT8)
├── data/                              # Jeux de données (train / test)
├── venv/                              # Environnement virtuel Python
├── .gitignore                         # Fichiers ignorés par Git
├── evaluate_cnn.py                    # Script d'évaluation et métriques de performance
├── gesture_detection.py               # Module de détection et d'interaction temps réel
├── main.py                            # Pipeline principal d'entraînement et exécution
├── models.py                          # Définition des architectures de réseaux de neurones
├── optimize_edge.py                   # Script d'optimisation Edge AI (Quantification PTQ)
├── prepare_data.py                    # Prétraitement et normalisation des échographies
├── pso_optimizer.py                   # Optimisation par essaim de particules (PSO) pour les hyperparamètres
├── README.md                          # Documentation du projet
└── requirements.txt                   # Dépendances et bibliothèques du projet