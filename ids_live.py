import os
import joblib
import pandas as pd
from datetime import datetime

# Ensure Scapy cache files are written to a local writable directory.
base_dir = os.path.dirname(__file__)
cache_dir = os.path.join(base_dir, '.scapy_cache')
os.makedirs(cache_dir, exist_ok=True)
os.environ.setdefault('XDG_CACHE_HOME', cache_dir)

from scapy.all import sniff, IP, TCP, UDP
from feature_extractor import extraire_caracteristiques

# =========================
# CONFIGURATION
# =========================
VICTIME_IP = "100.88.208.57"

print("⏳ Chargement des modèles IA et outils...")

model = joblib.load(os.path.join(base_dir, "ids_model_random_forest.pkl"))
scaler = joblib.load(os.path.join(base_dir, "scaler.pkl"))
pca = joblib.load(os.path.join(base_dir, "pca_transformer.pkl"))

le_protocol = joblib.load(os.path.join(base_dir, "label_encoder_protocol.pkl"))
le_service = joblib.load(os.path.join(base_dir, "label_encoder_service.pkl"))

print("✅ Modèles chargés avec succès.")


# =========================
# ANALYSE DES PAQUETS
# =========================
def analyser_observation(paquet):
    try:
        if paquet is None or not paquet.haslayer(IP):
            return

        ip_src = paquet[IP].src
        ip_dst = paquet[IP].dst

        # 🔥 On garde uniquement le trafic de la victime
        if ip_src != VICTIME_IP and ip_dst != VICTIME_IP:
            return

        features = extraire_caracteristiques(paquet, le_protocol, le_service)

        df = pd.DataFrame([features])
        X_scaled = scaler.transform(df)
        X_pca = pca.transform(X_scaled)

        prediction = model.predict(X_pca)

        if prediction[0] == 1:

            if paquet.haslayer(TCP):
                protocole = f"TCP (Port: {paquet[TCP].dport})"
            elif paquet.haslayer(UDP):
                protocole = f"UDP (Port: {paquet[UDP].dport})"
            else:
                protocole = "Autre"

            heure = datetime.now().strftime("%H:%M:%S")

            print(f"\n🚨 [ALERTE CRITIQUE - {heure}] 🚨")
            print(f"   🛑 IP Source      : {ip_src}")
            print(f"   🎯 IP Destination : {ip_dst}")
            print(f"   ⚙️ Protocole      : {protocole}")
            print(f"   ⚠️ Action         : TRAFIC SUSPECT DÉTECTÉ")
            print("-" * 60)

    except Exception as e:
        print(f"[Erreur IDS] {e}")


# =========================
# CAPTURE LIVE
# =========================
print("🛡️ IDS Démarré - Surveillance en direct...")
print(f"🎯 Victime surveillée : {VICTIME_IP}")

if __name__ == '__main__':
    sniff(
        prn=analyser_observation,
        store=False,
        filter=f"host {VICTIME_IP}"
    )
