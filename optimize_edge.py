import os
import time
import copy
import torch
import torch.nn as nn
from torchvision import models

def get_file_size_mb(path: str) -> float:
    """Calcule la taille d'un fichier sur disque en Megabytes (MB)."""
    if not os.path.exists(path):
        return 0.0
    return os.path.getsize(path) / (1024 * 1024)

def benchmark_inference(model: nn.Module, input_tensor: torch.Tensor, num_runs: int = 100) -> float:
    """Mesure la latence moyenne d'inférence sur CPU (scénario Edge/TinyML)."""
    model.eval()
    with torch.no_grad():
        for _ in range(10): # Warmup
            _ = model(input_tensor)
            
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)
    end_time = time.perf_counter()
    
    return ((end_time - start_time) / num_runs) * 1000.0

def quantize_model_weights_int8(model: nn.Module) -> nn.Module:
    """
    Quantification dynamique ciblée sur le modèle complet 
    adaptée au processeur hôte (CPU/Edge).
    """
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d},
        dtype=torch.qint8
    )
    return quantized_model

def optimize_and_benchmark():
    print("=================================================================")
    print("   OPTIMISATION EDGE AI / TINYML — QUANTIFICATION RESNET-18     ")
    print("=================================================================\n")
    
    device = torch.device("cpu")
    fp32_path = "checkpoints/model_resnet18.pth"
    int8_path = "checkpoints/model_resnet18_quantized.pth"
    
    # 1. Chargement du modèle ResNet-18 FP32
    model_fp32 = models.resnet18(weights=None)
    model_fp32.fc = nn.Linear(model_fp32.fc.in_features, 3)
    
    if os.path.exists(fp32_path):
        model_fp32.load_state_dict(torch.load(fp32_path, map_location=device))
        print(f"[+] Modèle FP32 d'origine chargé depuis : {fp32_path}")
    else:
        print(f"[!] {fp32_path} introuvable. Initialisation avec structure de base.")
    
    model_fp32.eval()
    
    # 2. Quantification du modèle
    print("[+] Application de la quantification INT8...")
    model_int8 = quantize_model_weights_int8(model_fp32)
    
    # Sauvegarde du dictionnaire de poids quantifiés
    torch.save(model_int8.state_dict(), int8_path)
    print(f"[+] Modèle quantifié INT8 sauvegardé sous : {int8_path}")

    # 3. Benchmark de performances (Latence & Empreinte mémoire)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    fp32_size = get_file_size_mb(fp32_path)
    int8_size = get_file_size_mb(int8_path)
    
    fp32_latency = benchmark_inference(model_fp32, dummy_input)
    int8_latency = benchmark_inference(model_int8, dummy_input)
    
    # 4. Synthèse des métriques pour le mémoire de Master
    print("\n-----------------------------------------------------------------")
    print("             RÉSULTATS DE COMPRESSION & BENCHMARK EMBARQUÉ        ")
    print("-----------------------------------------------------------------")
    print(f" Modèle FP32 (Original)   | Taille: {fp32_size:.2f} MB | Latence: {fp32_latency:.2f} ms")
    print(f" Modèle INT8 (Optimisé)   | Taille: {int8_size:.2f} MB | Latence: {int8_latency:.2f} ms")
    
    if int8_size > 0 and fp32_size > 0:
        compression = fp32_size / int8_size
        speedup = fp32_latency / int8_latency
        print("-----------------------------------------------------------------")
        print(f" Facteur de réduction mémoire : {compression:.2f}x")
        print(f" Gain d'accélération (Speedup): {speedup:.2f}x")
    print("-----------------------------------------------------------------\n")

if __name__ == "__main__":
    optimize_and_benchmark()