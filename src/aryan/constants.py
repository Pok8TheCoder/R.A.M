"""Trimmed constants ported from Aryan's `origin/aryan:src/utils/constants.py`.

Only the pieces `src/aryan/world_model.py` and the forecast-player session
builder actually need -- MITRE stage count/names for the classification head
and label bookkeeping.
"""

MITRE_STAGES = {
    "Benign": 0,
    "Reconnaissance": 1,
    "Initial Access": 2,
    "Lateral Movement": 3,
    "Command & Control": 4,
    "Exfiltration": 5,
    "Impact": 6,
}

MITRE_STAGES_INV = {v: k for k, v in MITRE_STAGES.items()}

NUM_MITRE_STAGES = len(MITRE_STAGES)

CICIDS_LABEL_TO_MITRE = {
    "Benign": "Benign",
    "FTP-BruteForce": "Initial Access",
    "SSH-Bruteforce": "Initial Access",
    "Bot": "Command & Control",
    "Infilteration": "Lateral Movement",
    "DoS attacks-Hulk": "Impact",
    "DoS attacks-SlowHTTPTest": "Impact",
    "DoS attacks-Slowloris": "Impact",
    "DoS attacks-GoldenEye": "Impact",
    "DDoS attacks-LOIC-HTTP": "Impact",
    "DDOS attack-HOIC": "Impact",
    "DDOS attack-LOIC-UDP": "Impact",
    "Brute Force -Web": "Initial Access",
    "Brute Force -XSS": "Initial Access",
    "SQL Injection": "Initial Access",
}

# One canonical CIC-IDS-2018 attack name per MITRE stage head — what ARY.01/02
# were trained to distinguish (not abstract tactic names like "Lateral Movement").
STAGE_ID_TO_ATTACK = {
    0: "Benign",
    1: "Network Service Scanning",
    2: "SSH-Bruteforce",
    3: "Infilteration",
    4: "Bot",
    5: "Exfiltration",
    6: "DoS attacks-Hulk",
}

# Kill-chain demo splice: real held-out windows keyed by stage id.
KILLCHAIN_ATTACKS = {
    2: "SSH-Bruteforce",
    3: "Infilteration",
    6: "DoS attacks-Hulk",
}

# Catalog primary_tactic strings -> Aryan 7-stage names
TACTIC_TO_STAGE = {
    "Normal": "Benign",
    "Reconnaissance": "Reconnaissance",
    "Initial Access": "Initial Access",
    "Execution": "Initial Access",
    "Persistence": "Lateral Movement",
    "Privilege Escalation": "Lateral Movement",
    "Defense Evasion": "Lateral Movement",
    "Credential Access": "Lateral Movement",
    "Discovery": "Reconnaissance",
    "Lateral Movement": "Lateral Movement",
    "Collection": "Exfiltration",
    "Command and Control": "Command & Control",
    "Command & Control": "Command & Control",
    "Exfiltration": "Exfiltration",
    "Impact": "Impact",
}
