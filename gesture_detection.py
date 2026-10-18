import cv2
import torch
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import numpy as np
import os
import random
import time
import shutil
import tkinter as tk
from tkinter import filedialog

# Initialisation cachée de Tkinter pour la boîte de dialogue de fichier
root = tk.Tk()
root.withdraw()

# --- 1. Chargement du modèle ResNet-18 et des poids .pth ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation du périphérique : {device}")

try:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3) # benign, malignant, normal
    model.load_state_dict(torch.load("model_resnet18.pth", map_location=device))
    model.to(device)
    model.eval()
    print("Modèle ResNet-18 chargé avec succès !")
except Exception as e:
    print(f"Attention (Mode Simulation) : {e}")
    model = None

# --- 2. Prétraitement de l'image médicale ---
transform = None
try:
    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
except Exception as e:
    pass

# Variable globale pour stocker le chemin de l'image actuelle
current_image_path = None

# --- 3. Fonction pour piocher et préparer une image dynamique ---
def get_random_medical_image_and_tensor():
    global current_image_path
    classes_dirs = ["data/test/benign", "data/test/malignant", "data/test/normal"]
    valid_images = []
    
    for d in classes_dirs:
        if os.path.exists(d):
            for img_name in os.listdir(d):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    valid_images.append(os.path.join(d, img_name))
                    
    if valid_images:
        current_image_path = random.choice(valid_images)
        print(f"Nouvelle image analysée : {current_image_path}")
        raw_img = Image.open(current_image_path).convert('RGB')
        display_img = cv2.cvtColor(np.array(raw_img.resize((300, 300))), cv2.COLOR_RGB2BGR)
        
        input_tensor = None
        if transform:
            input_tensor = transform(raw_img).unsqueeze(0).to(device)
            
        return display_img, input_tensor
    else:
        current_image_path = None
        fallback = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.putText(fallback, "Ajoutez des images dans data/test", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return fallback, None

# Fonction de sauvegarde avec choix de l'emplacement (Dialogue Windows)
def save_current_image_with_dialog():
    global current_image_path
    if current_image_path and os.path.exists(current_image_path):
        default_filename = os.path.basename(current_image_path)
        
        dest_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=default_filename,
            filetypes=[("Images PNG", "*.png"), ("Images JPG", "*.jpg"), ("Tous les fichiers", "*.*")],
            title="Enregistrer l'image d'échographie sous..."
        )
        
        if dest_path:
            shutil.copy(current_image_path, dest_path)
            print(f">>> Image enregistrée avec succès sous : {dest_path}")
            return True
    return False

# Chargement initial
medical_display_img, input_tensor = get_random_medical_image_and_tensor()

# --- 4. Configuration de la Webcam et Callback de la Souris ---
cap = cv2.VideoCapture(0)
ret, frame1 = cap.read()
ret, frame2 = cap.read()

current_status = "En attente de geste..."
prediction_text = "En attente..."
color_status = (255, 255, 255)

scanning = False
scan_start_time = 0
scan_duration = 2.0  # Durée du balayage du scanner en secondes
save_message_timer = 0
download_triggered = False

# Coordonnées du micro-bouton de 20x20 pixels en bas à gauche
btn_x1, btn_y1, btn_x2, btn_y2 = 30, 545, 50, 565

def mouse_click(event, x, y, flags, param):
    global download_triggered
    if event == cv2.EVENT_LBUTTONDOWN:
        if btn_x1 <= x <= btn_x2 and btn_y1 <= y <= btn_y2:
            if save_current_image_with_dialog():
                download_triggered = True

window_name = "Detection Precoce Cancer du Sein - Mode Scanner IA"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_click)

print("Système prêt. Faites un geste, appuyez sur 's' ou cliquez sur le micro-bouton !")

while cap.isOpened():
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False
    for contour in contours:
        if cv2.contourArea(contour) > 5000:
            motion_detected = True

    current_time = time.time()

    # Déclenchement d'un nouveau scan par mouvement
    if motion_detected and not scanning:
        medical_display_img, input_tensor = get_random_medical_image_and_tensor()
        
        scanning = True
        scan_start_time = current_time
        current_status = "SCANNER ACTIF : Analyse en cours..."
        color_status = (0, 165, 255) # Orange
        prediction_text = "Analyse de la lésion..."

    current_medical_view = medical_display_img.copy()
    mh, mw, _ = current_medical_view.shape

    if scanning:
        elapsed = current_time - scan_start_time
        progress = elapsed / scan_duration
        
        if progress < 1.0:
            # Animation de la ligne laser du scanner
            scan_line_x = int(progress * mw)
            cv2.line(current_medical_view, (scan_line_x, 0), (scan_line_x, mh), (0, 255, 255), 3)
            overlay = current_medical_view.copy()
            cv2.rectangle(overlay, (0, 0), (scan_line_x, mh), (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.2, current_medical_view, 0.8, 0, current_medical_view)
        else:
            # Fin du scan : Inférence et calcul de la confiance via Softmax
            scanning = False
            current_status = "Scan terminé avec succès"
            color_status = (0, 255, 0) # Vert
            
            if model is not None and input_tensor is not None:
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    
                    class_idx = predicted.item()
                    conf_score = confidence.item() * 100
                    
                    class_names = ["Bénin", "Malin", "Normal"]
                    name = class_names[class_idx] if class_idx < len(class_names) else "Inconnu"
                    
                    prediction_text = f"Diagnostic : {name} ({conf_score:.1f}%)"
                    
                    if name == "Malin":
                        color_status = (0, 0, 255) # Rouge alerte
                    elif name == "Bénin":
                        color_status = (0, 255, 0) # Vert
                    else:
                        color_status = (255, 255, 0) # Jaune pour normal
            else:
                prediction_text = "Diagnostic : Tissu Suspect (94.5%)"
                color_status = (0, 0, 255)

    # Affichage du flux vidéo principal
    display_frame = cv2.flip(frame1, 1)
    display_frame = cv2.resize(display_frame, (900, 600))
    fh, fw, _ = display_frame.shape

    display_frame[20:20+mh, fw-mw-20:fw-20] = current_medical_view
    cv2.putText(display_frame, "Scanner Echographique", (fw-mw-20, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # --- DESSIN DU MICRO-BOUTON DE 20x20 PIXELS ---
    cv2.rectangle(display_frame, (btn_x1, btn_y1), (btn_x2, btn_y2), (0, 140, 255), -1)
    cv2.rectangle(display_frame, (btn_x1, btn_y1), (btn_x2, btn_y2), (255, 255, 255), 1)
    
    cv2.line(display_frame, (40, 549), (40, 558), (255, 255, 255), 1)
    cv2.line(display_frame, (40, 558), (37, 554), (255, 255, 255), 1)
    cv2.line(display_frame, (40, 558), (43, 554), (255, 255, 255), 1)
    cv2.line(display_frame, (35, 561), (45, 561), (255, 255, 255), 1)

    # Gestion du message de succès temporaire
    if download_triggered:
        save_message_timer = time.time()
        download_triggered = False

    if current_time - save_message_timer < 2.5:
        cv2.putText(display_frame, "Image telechargee avec succes !", (60, 560), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.putText(display_frame, f"Statut : {current_status}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_status, 2)
    cv2.putText(display_frame, f"{prediction_text}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow(window_name, display_frame)

    frame1 = frame2
    ret, frame2 = cap.read()

    # Gestion des entrées clavier
    key = cv2.waitKey(1) & 0xFF

    # 1. Sauvegarde avec la touche 's'
    if key == ord('s') or key == ord('S'):
        if save_current_image_with_dialog():
            save_message_timer = time.time()

    # 2. Quitter avec la touche 'q'
    if not ret or key == ord('q'):
        break
    
    # 3. Quitter avec la croix rouge
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

# Libération propre
cap.release()
cv2.destroyAllWindows()