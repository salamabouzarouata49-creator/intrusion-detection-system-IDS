import os

# Ensure Scapy cache files are written to a local writable directory.
cache_dir = os.path.join(os.path.dirname(__file__), '.scapy_cache')
os.makedirs(cache_dir, exist_ok=True)
os.environ.setdefault('XDG_CACHE_HOME', cache_dir)

from scapy.all import IP, TCP, UDP, ICMP

def extraire_caracteristiques(paquet, le_protocol, le_service):
    """
    Extrait les caractéristiques d'un paquet Scapy et applique l'encodage
    pour correspondre au format attendu par le modèle NSL-KDD.
    """
    if paquet is None:
        return None

    # Valeurs par défaut (variables sous forme de texte avant encodage)
    proto_str = "tcp"
    service_str = "private"
    dst_bytes = 0

    # Analyse basique du paquet Scapy
    if paquet.haslayer(IP):
        dst_bytes = len(paquet[IP].payload)
        
        if paquet.haslayer(TCP):
            proto_str = "tcp"
            port = paquet[TCP].dport
            # Mapping très simplifié des services
            if port == 80: service_str = "http"
            elif port in [20, 21]: service_str = "ftp"
            elif port == 22: service_str = "ssh"
            elif port == 23: service_str = "telnet"
        elif paquet.haslayer(UDP):
            proto_str = "udp"
            if paquet[UDP].dport == 53: service_str = "domain_u"
        elif paquet.haslayer(ICMP):
            proto_str = "icmp"

    # Encodage sécurisé (si le protocole/service est inconnu, on met 0)
    try:
        proto_encoded = le_protocol.transform([proto_str])[0]
    except ValueError:
        proto_encoded = 0

    try:
        service_encoded = le_service.transform([service_str])[0]
    except ValueError:
        service_encoded = 0

    # Dictionnaire final avec les 19 caractéristiques exactes
    features = {
        "protocol_type": proto_encoded,
        "service": service_encoded,
        "logged_in": 0,
        "is_host_login": 0,
        "is_guest_login": 0,
        "dst_bytes": dst_bytes,
        "num_shells": 0,
        "num_outbound_cmds": 0,
        "srv_count": 0,
        "diff_srv_rate": 0,
        "srv_diff_host_rate": 0,
        "dst_host_count": 0,
        "dst_host_srv_count": 0,
        "dst_host_same_srv_rate": 0,
        "dst_host_diff_srv_rate": 0,
        "dst_host_same_src_port_rate": 0,
        "dst_host_srv_diff_host_rate": 0,
        "dst_host_serror_rate": 0,
        "dst_host_srv_serror_rate": 0
    }

    return features