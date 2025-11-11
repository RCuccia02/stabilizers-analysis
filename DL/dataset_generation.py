import numpy as np
import matplotlib.pyplot as plt

# --- 1. Parametri di Generazione (Sintetico) ---
NUM_SAMPLES = 1000  
SEQ_LENGTH = 500    
MAX_PAN_SPEED = 0.5 
JITTER_STRENGTH = 1.5 
SHOCK_PROBABILITY = 0.01  # 1% di probabilità per ogni frame di avere uno shock
SHOCK_STRENGTH = 8.0      # Quanto è grande il salto

t = np.linspace(0, 10, SEQ_LENGTH)

X_act_data = []  
X_smooth_data = [] 

# --- 2. Ciclo di Generazione (Sintetico con Shock) ---
print(f"Generazione di {NUM_SAMPLES} campioni (con Jitter e Shock)...")

for _ in range(NUM_SAMPLES):
    
    # FASE A: Crea X_smooth (Il panning pulito)
    f1 = np.random.uniform(0.1, MAX_PAN_SPEED)
    f2 = np.random.uniform(0.1, MAX_PAN_SPEED)
    a1 = np.random.uniform(0.5, 2.0)
    a2 = np.random.uniform(0.5, 2.0)
    p1 = np.random.uniform(0, np.pi)
    p2 = np.random.uniform(0, np.pi)
    X_smooth = a1 * np.sin(2 * np.pi * f1 * t + p1) + \
               a2 * np.sin(2 * np.pi * f2 * t + p2)
    
    # FASE B: Aggiungi il Jitter
    noise = np.random.normal(0, JITTER_STRENGTH, SEQ_LENGTH)
    X_act = X_smooth + noise
    
    # FASE C: Aggiungi Shock (Fallimenti Tracciamento)
    num_shocks = int(SEQ_LENGTH * SHOCK_PROBABILITY)
    for _ in range(num_shocks):
        shock_idx = np.random.randint(0, SEQ_LENGTH)
        shock_value = np.random.uniform(-SHOCK_STRENGTH, SHOCK_STRENGTH)
        X_act[shock_idx:] += shock_value
    
    X_act_data.append(X_act)
    X_smooth_data.append(X_smooth)

print("Generazione completata.")

X_train = np.array(X_act_data)
y_train = np.array(X_smooth_data)


# ==========================================================
# --- 3. Caricamento Dati Reali e Plotting (Come Prima) ---
# ==========================================================

# --- 3A: Carica i dati REALI ---
file_reale = "traiettoria_rumorosa_X_act.npy"
try:
    trajectory_array = np.load(file_reale)
    # Plottiamo l'asse X
    real_data_x = trajectory_array[:, 0] 
    t_real = np.arange(len(real_data_x))
    print(f"Dati reali '{file_reale}' caricati: {trajectory_array.shape}")
    dati_reali_caricati = True
except FileNotFoundError:
    print(f"--- ATTENZIONE: File '{file_reale}' non trovato. Mostro solo il sintetico. ---")
    dati_reali_caricati = False

# --- 3B: Crea 2 Subplot (uno sopra l'altro) ---
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))

# --- Plot 1: Dati Sintetici (CON GLI SHOCK) ---
sample_idx = 0
ax1.set_title("Esempio dal Dataset Sintetico (Ora con Shock)")
ax1.plot(t, X_act_data[sample_idx], label="X_act (Sintetico Rumoroso + Shock)", 
         alpha=0.7, linestyle='--')
ax1.plot(t, X_smooth_data[sample_idx], label="X_smooth (Sintetico Pulito)", 
         linewidth=3, color='black')
ax1.set_xlabel("Tempo (frame)")
ax1.set_ylabel("Posizione (Sintetica)")
ax1.legend()
ax1.grid(True)

# --- Plot 2: Dati Reali (IL GRAFICO ROSSO) ---
ax2.set_title("Dati Reali da 'traiettoria_rumorosa_X_act.npy'")
if dati_reali_caricati:
    ax2.plot(t_real, real_data_x, label="X_act (Reale, asse X)", 
             color='red')
ax2.set_xlabel("Tempo (frame)")
ax2.set_ylabel("Posizione (Pixel Accumulati)")
ax2.legend()
ax2.grid(True)

plt.tight_layout() 
plt.show()