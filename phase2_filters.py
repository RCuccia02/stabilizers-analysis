import numpy as np
import os
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter

# --- Funzioni Helper Interne ---

def _crea_filtro_kalman_1D(R_val, Q_val):
    """
    Helper per creare un filtro di Kalman 1D
    con modello a velocità costante.
    """
    # Crea il filtro
    kf = KalmanFilter(dim_x=2, dim_z=1)

    # Definisci il Modello di Stato (Come si evolve lo stato)
    dt = 1.0 # (frame per frame)
    kf.F = np.array([[1., dt],
                     [0., 1.]])
    
    # Definisci il Modello di Misura (Come misuriamo lo stato)
    kf.H = np.array([[1., 0.]])
    
    # Definisci i Rumori (L'INCERTEZZA)
    # Q = Rumore di Processo (Quanto ci fidiamo del modello a vel. costante?)
    #     Valore alto = il modello è inaffidabile (la velocità cambia spesso)
    kf.Q *= Q_val
    
    # R = Rumore di Misura (Quanto ci fidiamo del nostro X_act.npy?)
    #     Valore alto = i nostri dati di Fase 1 sono spazzatura
    kf.R *= R_val

    # Definisci lo stato iniziale
    kf.x = np.zeros((2, 1))
    
    return kf

def _filter_fps_cutoff(signal, cutoff):
    # Implementazione "Cutoff" con taglio netto
    signal_freq = np.fft.fft(signal) # Trasformata di Fourier
    n = len(signal)
    filtered_freq = signal_freq.copy()
    cut_idx = int(n * cutoff) # Indice di cutoff
    filtered_freq[cut_idx:-cut_idx] = 0 # Azzeramento frequenze alte (si trovano in mezzo)
    signal_filtered = np.fft.ifft(filtered_freq).real # Inversa e parte reale
    return signal_filtered

def _filter_fps_gaussian(signal, sigma):
    # Implementazione "Gaussian"
    n = len(signal)
    if n == 0: 
        return np.array([])
    signal_freq = np.fft.fft(signal)
    freqs = np.fft.fftfreq(n)
    if sigma == 0: 
        return signal
    gauss_window = np.exp(- (freqs**2) / (2 * (sigma**2))) # Finestra gaussiana
    filtered_freq = signal_freq * gauss_window # Applica la finestra
    signal_filtered = np.fft.ifft(filtered_freq).real
    return signal_filtered

def _plot_results(t, X_act_x, X_lpf_x, X_act_y, X_lpf_y, X_act_theta, X_lpf_theta, title):
    # Genera un grafico comparativo tra X_act e X_smooth
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(14, 12), sharex=True)
    fig.suptitle(title, fontsize=16)

    ax1.set_title("Asse X")
    ax1.plot(t, X_act_x, label="X_act(n) (Rumoroso/Reale)", color='red', alpha=0.7)
    ax1.plot(t, X_lpf_x, label="X_smooth(n) (Filtrato)", color='black', linewidth=2)
    ax1.legend()
    ax1.grid(True)

    ax2.set_title("Asse Y")
    ax2.plot(t, X_act_y, label="X_act(n) (Rumoroso/Reale)", color='blue', alpha=0.7)
    ax2.plot(t, X_lpf_y, label="X_smooth(n) (Filtrato)", color='black', linewidth=2)
    ax2.legend()
    ax2.grid(True)

    ax3.set_title("Asse Theta (Angolo)")
    ax3.plot(t, X_act_theta, label="Theta_act(n) (Rumoroso/Reale)", color='green', alpha=0.7)
    ax3.plot(t, X_lpf_theta, label="Theta_smooth(n) (Filtrato)", color='black', linewidth=2)
    ax3.legend()
    ax3.grid(True)

    plt.xlabel("Tempo (frame)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(f"./images/plots/{title.replace(' ', '_')}.png")

# --- Funzioni Principali ---

def run_fps_filter(x_act_path, output_dir, video_name, smoothing_method, sigma, cutoff):
    """
    Carica X_act, applica il filtro FPS selezionato e salva il risultato.
    Ritorna il percorso al file X_smooth.
    """
    print(f"--- Avvio Fase 2: Filtro FPS ({smoothing_method}) ---")
    
    try:
        X_act = np.load(x_act_path)
    except FileNotFoundError:
        print(f"ERRORE (Fase 2 - FPS): File non trovato {x_act_path}")
        return None

    X_act_x = X_act[:, 0]
    X_act_y = X_act[:, 1]
    X_act_theta = X_act[:, 2]

    if smoothing_method == "cutoff":
        print(f"Applicazione Filtro FPS (Cutoff={cutoff})...")
        X_lpf_x = _filter_fps_cutoff(X_act_x, cutoff)
        X_lpf_y = _filter_fps_cutoff(X_act_y, cutoff)
        X_lpf_theta = _filter_fps_cutoff(X_act_theta, cutoff)
    elif smoothing_method == "gaussian":
        print(f"Applicazione Filtro FPS (Gaussiano, Sigma={sigma})...")
        X_lpf_x = _filter_fps_gaussian(X_act_x, sigma)
        X_lpf_y = _filter_fps_gaussian(X_act_y, sigma)
        X_lpf_theta = _filter_fps_gaussian(X_act_theta, sigma)
    else:
        raise ValueError("Metodo di smoothing non riconosciuto.")

    X_lpf_array = np.stack([X_lpf_x, X_lpf_y, X_lpf_theta], axis=1)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_file = os.path.join(output_dir, f"X_smooth_FPS_{smoothing_method}_{video_name}.npy")
    np.save(output_file, X_lpf_array)
    print(f"Fase 2 (FPS) completata. Salvato in: {output_file}")
    
    # Genera grafico
    t_real = np.arange(len(X_act_x))
    _plot_results(t_real, X_act_x, X_lpf_x, X_act_y, X_lpf_y, X_act_theta, X_lpf_theta, 
                  f"Risultati Filtro FPS ({smoothing_method.upper()})")

    return output_file

def run_mvi_filter(v_act_path, x_act_path, output_dir, video_name, delta=0.9):
    """
    Implementa il filtro MVI ESATTAMENTE come da slide 11 e 12 [cite: 96-115].
    Ritorna il percorso al file X_smooth.
    """
    print(f"--- Avvio Fase 2: Filtro MVI (Metodo Prof., Delta={delta}) ---")
    
    try:
        V_act = np.load(v_act_path)
        X_initial = np.load(x_act_path)
    except FileNotFoundError:
        print("ERRORE (Fase 2 - MVI): File .npy non trovati.")
        return None

    if len(V_act) != len(X_initial):
        print("ERRORE (MVI): V_act e X_act non sono sincronizzati!")
        return None
        
    n_frames = len(V_act)
    V_int = np.zeros_like(V_act) 
    X_smooth_MVI = np.zeros_like(X_initial)

    V_int[0] = V_act[0] 
    X_smooth_MVI[0] = X_initial[0]

    for n in range(1, n_frames):
        V_int[n] = (delta * V_int[n-1]) + V_act[n] # Motion Vector tra frame n-1 e n
        
        X_smooth_MVI[n] = X_initial[n] - V_int[n] # Posizione stabilizzata a frame n
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_file = os.path.join(output_dir, f"X_smooth_MVI_{video_name}.npy")
    np.save(output_file, X_smooth_MVI)
    print(f"Fase 2 (MVI) completata. Salvato in: {output_file}")
    
    # Genera grafico
    t_real = np.arange(len(X_initial))
    _plot_results(t_real, X_initial[:,0], X_smooth_MVI[:,0], 
                  X_initial[:,1], X_smooth_MVI[:,1], 
                  X_initial[:,2], X_smooth_MVI[:,2], 
                  f"Risultati Filtro MVI (Delta={delta})")
    
    return output_file

def run_kalman_filter(x_act_path, output_dir, video_name, R_val=10.0, Q_val=0.001):
    """
    Carica X_act, applica il filtro di Kalman e salva il risultato X_smooth.
    Ritorna il percorso al file X_smooth.    
    """
    print(f"--- Avvio Fase 2: Filtro Kalman (R={R_val}, Q={Q_val}) ---")
    
    try:
        X_act = np.load(x_act_path)
    except FileNotFoundError:
        print(f"ERRORE (Fase 2 - Kalman): File non trovato {x_act_path}")
        return None
        
    n_frames = len(X_act)
    
    # Creazione dei filtri
    kf_x = _crea_filtro_kalman_1D(R_val=R_val, Q_val=Q_val)
    kf_y = _crea_filtro_kalman_1D(R_val=R_val, Q_val=Q_val)
    kf_theta = _crea_filtro_kalman_1D(R_val=R_val, Q_val=Q_val)

    # Array per salvare lo stato filtrato
    X_smooth_Kalman = np.zeros_like(X_act)

    for n in range(n_frames):
        kf_x.predict()
        kf_y.predict()
        kf_theta.predict()

        kf_x.update(X_act[n, 0])
        kf_y.update(X_act[n, 1])
        kf_theta.update(X_act[n, 2])
        
        X_smooth_Kalman[n, 0] = kf_x.x[0, 0]
        X_smooth_Kalman[n, 1] = kf_y.x[0, 0]
        X_smooth_Kalman[n, 2] = kf_theta.x[0, 0]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_file = os.path.join(output_dir, f"X_smooth_Kalman_{video_name}.npy")
    np.save(output_file, X_smooth_Kalman)
    print(f"Fase 2 (Kalman) completata. Salvato in: {output_file}")
    
    # Genera grafico
    t_real = np.arange(n_frames)
    _plot_results(t_real, X_act[:,0], X_smooth_Kalman[:,0], 
                  X_act[:,1], X_smooth_Kalman[:,1], 
                  X_act[:,2], X_smooth_Kalman[:,2], 
                  f"Risultati Filtro Kalman (R={R_val}, Q={Q_val})")
    
    return output_file