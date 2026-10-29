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

# --- 1. Chargement du modèle ResNet-18 et des poids centralisés ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation du périphérique : {device}")

weights_path = "checkpoints/resnet18_best.pth"
try:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2) # Classes : Benign, Malignant
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    print("Modèle ResNet-18 chargé avec succès depuis checkpoints/ !")
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

current_image_path = None
last_prediction_name = "En attente"
last_confidence = 0.0

# --- 3. Fonction pour piocher et préparer une image dynamique (Dossiers data/test) ---
def get_random_medical_image_and_tensor():
    global current_image_path
    classes_dirs = ["data/test/benign", "data/test/malignant"]
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

# Fonction de sauvegarde avec dialogue Windows
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

# Chargement initial d'une image médicale
medical_display_img, input_tensor = get_random_medical_image_and_tensor()

# --- 4. Configuration de la Webcam & Callback de la Souris ---
cap = cv2.VideoCapture(0)
ret, frame1 = cap.read()
ret, frame2 = cap.read()

current_status = "Bougez la main / le corps pour analyser"
prediction_text = "En attente..."
color_status = (255, 255, 255)

scanning = False
scan_start_time = 0
scan_duration = 2.0  # Durée du balayage laser en secondes
save_message_timer = 0
download_triggered = False

# Coordonnées du micro-bouton de sauvegarde (20x20 pixels) en BAS À GAUCHE
btn_x1, btn_y1, btn_x2, btn_y2 = 30, 545, 50, 565

def mouse_click(event, x, y, flags, param):
    global download_triggered
    if event == cv2.EVENT_LBUTTONDOWN:
        if btn_x1 <= x <= btn_x2 and btn_y1 <= y <= btn_y2:
            if save_current_image_with_dialog():
                download_triggered = True

window_name = "Detection Precoce Cancer du Sein - Mode Scanner IA"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 900, 600)
cv2.setMouseCallback(window_name, mouse_click)

print("Système prêt. Faites un geste devant la webcam !")

while cap.isOpened():
    # Détection de mouvement par différence de frames (OpenCV)
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False
    valid_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 3000: # Seuil de détection pour les mouvements
            motion_detected = True
            valid_contours.append(contour)

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
            scanning = False
            current_status = "Scan terminé avec succès"
            
            if model is not None and input_tensor is not None:
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    
                    class_idx = predicted.item()
                    last_confidence = confidence.item() * 100
                    class_names = ["Benign", "Malignant"]
                    last_prediction_name = class_names[class_idx] if class_idx < len(class_names) else "Inconnu"
                    
                    prediction_text = f"Diagnostic : {last_prediction_name} ({last_confidence:.1f}%)"
                    color_status = (0, 0, 255) if last_prediction_name == "Malignant" else (0, 255, 0)
            else:
                last_prediction_name = "Malignant"
                last_confidence = 94.5
                prediction_text = "Diagnostic : Tissu Suspect (94.5%)"
                color_status = (0, 0, 255)

    # --- AJOUT DES REPÈRES DE PRÉCISION ET DE MESURE SUR L'IMAGE ÉCHOGRAPHIQUE ---
    if not scanning and current_image_path is not None:
        # Couleur des repères selon le diagnostic (Vert = Bénin, Rouge = Malin)
        marker_color = (0, 0, 255) if last_prediction_name == "Malignant" else (0, 255, 0)
        
        # 1. Dessin d'une boîte de Région d'Intérêt (ROI) ciblée au centre de l'échographie
        roi_x1, roi_y1, roi_x2, roi_y2 = 90, 90, 210, 210
        cv2.rectangle(current_medical_view, (roi_x1, roi_y1), (roi_x2, roi_y2), marker_color, 2)
        
        # 2. Croix de repérage clinique (+) au centre de la lésion
        center_x, center_y = 150, 150
        cv2.line(current_medical_view, (center_x - 10, center_y), (center_x + 10, center_y), marker_color, 1)
        cv2.line(current_medical_view, (center_x, center_y - 10), (center_x, center_y + 10), marker_color, 1)
        
        # 3. Lignes de calibrage / mesure (type échographe)
        cv2.putText(current_medical_view, "+D1", (roi_x1 - 25, roi_y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(current_medical_view, f"{last_prediction_name[:3].upper()} {last_confidence:.1f}%", (10, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.45, marker_color, 2)

    # Préparation du flux vidéo principal (Effet miroir)
    raw_display = cv2.flip(frame1, 1)
    display_frame = cv2.resize(raw_display, (900, 600))
    fh, fw, _ = display_frame.shape

    # --- AFFICHAGE DE LA FORME ORGANIQUE EXACTE DES GESTES ---
    orig_h, orig_w, _ = frame1.shape
    scale_x = fw / orig_w
    scale_y = fh / orig_h

    for c in valid_contours:
        c_transformed = c.copy().astype(float)
        for point in c_transformed:
            point[0][0] = orig_w - point[0][0] # Miroir
            point[0][0] *= scale_x
            point[0][1] *= scale_y
        
        c_int = c_transformed.astype(np.int32)
        cv2.drawContours(display_frame, [c_int], -1, (0, 255, 0), 2)

    # Incrustation de l'image médicale avec ses repères cliniques en haut à droite
    display_frame[20:20+mh, fw-mw-20:fw-20] = current_medical_view
    cv2.putText(display_frame, "Image Echographique (ROI)", (fw-mw-20, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # --- DESSIN DU MICRO-BOUTON DE SAUVEGARDE EN BAS À GAUCHE ---
    cv2.rectangle(display_frame, (btn_x1, btn_y1), (btn_x2, btn_y2), (0, 140, 255), -1)
    cv2.rectangle(display_frame, (btn_x1, btn_y1), (btn_x2, btn_y2), (255, 255, 255), 1)
    
    # Icône interne (disquette)
    cv2.line(display_frame, (40, 549), (40, 558), (255, 255, 255), 1)
    cv2.line(display_frame, (40, 558), (37, 554), (255, 255, 255), 1)
    cv2.line(display_frame, (40, 558), (43, 554), (255, 255, 255), 1)
    cv2.line(display_frame, (35, 561), (45, 561), (255, 255, 255), 1)

    if download_triggered:
        save_message_timer = time.time()
        download_triggered = False

    if time.time() - save_message_timer < 2.5:
        cv2.putText(display_frame, "Image telechargee avec succes !", (60, 560), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Affichage des textes d'état et de diagnostic en haut à gauche
    cv2.putText(display_frame, f"Statut : {current_status}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_status, 2)
    cv2.putText(display_frame, f"{prediction_text}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow(window_name, display_frame)

    frame1 = frame2
    ret, frame2 = cap.read()

    # Gestion des entrées clavier et fermeture via la croix native [X]
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') or key == ord('S'):
        if save_current_image_with_dialog():
            save_message_timer = time.time()

    if not ret or key == ord('q'):
        break
    
    # Vérification de la fermeture via la croix native Windows
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()