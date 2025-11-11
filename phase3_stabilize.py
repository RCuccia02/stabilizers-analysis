import numpy as np
import cv2
import os

def run_phase3(video_input_path, x_act_path, x_smooth_path, output_video_path, trim_config={}):
    """
    Esegue la Fase 3: Stabilizzazione, Cropping e Trimming.
    Crea il video finale stabilizzato.
    
    Ritorna True se ha successo, False altrimenti.
    """
    print(f"--- Avvio Fase 3: Stabilizzazione per {output_video_path} ---")
    
    # Trimming
    TRIM_START_FRAMES = trim_config.get("start", 0)
    TRIM_END_FRAMES = trim_config.get("end", 0)

    try:
        X_act = np.load(x_act_path)
        X_smooth = np.load(x_smooth_path)
    except FileNotFoundError:
        print("--- FALLIMENTO CRITICO (Fase 3) ---")
        print("File traiettoria non trovati. Controllare i percorsi:")
        print(f"  X_act: {x_act_path}")
        print(f"  X_smooth: {x_smooth_path}")
        return False

    cap = cv2.VideoCapture(video_input_path)
    if not cap.isOpened():
        print(f"ERRORE (Fase 3): Impossibile aprire {video_input_path}")
        return False

    # Informazioni video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    output_dir = os.path.dirname(output_video_path)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
        
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    print(f"Dimensioni video: {frame_width}x{frame_height}")

    # Calcolo dei bordi da nascondere
    min_len = min(len(X_act), len(X_smooth))
    X_act_sync = X_act[:min_len]
    X_smooth_sync = X_smooth[:min_len]

    dx_corr = X_smooth_sync[:, 0] - X_act_sync[:, 0]
    dy_corr = X_smooth_sync[:, 1] - X_act_sync[:, 1]
    d_theta_corr = X_smooth_sync[:, 2] - X_act_sync[:, 2]

    start_idx = TRIM_START_FRAMES
    end_idx = len(dx_corr) - TRIM_END_FRAMES
    if end_idx <= start_idx:
        print("ATTENZIONE: Trimming troppo aggressivo. Analizzo tutti i frame.")
        start_idx = 0
        end_idx = len(dx_corr)

    if start_idx >= end_idx:
        print("ERRORE: Il video è troppo corto per i parametri di trimming!")
        cap.release()
        out.release()
        return False

    print(f"Analisi bordi eseguita solo sui frame {start_idx}-{end_idx}")

    # Calcola i massimi delle correzioni
    safe_dx_corr = dx_corr[start_idx:end_idx]
    safe_dy_corr = dy_corr[start_idx:end_idx]

    max_dx = np.max(np.abs(safe_dx_corr))
    max_dy = np.max(np.abs(safe_dy_corr))

    print("Analisi completata (Regione Sicura):")
    print(f"  Correzione Massima X: +/- {max_dx:.2f} pixel")
    print(f"  Correzione Massima Y: +/- {max_dy:.2f} pixel")

    # Calcola il fattore di zoom per nascondere i bordi
    border_x = int(np.ceil(max_dx))
    border_y = int(np.ceil(max_dy))

    scale_x = (frame_width - 2 * border_x) / frame_width
    scale_y = (frame_height - 2 * border_y) / frame_height
    scale_factor = min(scale_x, scale_y) # La scala dell'area "sicura"

    if scale_factor <= 0.01: # Buffer per evitare zoom infiniti/negativi
        print(f"ATTENZIONE: Correzioni ({max_dx}, {max_dy}) troppo grandi. Lo zoom sarà estremo.")
        zoom_factor = 20.0 # Limite massimo di zoom
    else:
        zoom_factor = 1.0 / scale_factor

    print(f"Applicazione zoom: {zoom_factor*100:.2f}% per nascondere i bordi.")

    # Matrice di trasformazione per lo zoom e il centraggio
    M_zoom = np.zeros((2, 3), dtype=np.float32)
    M_zoom[0, 0] = zoom_factor
    M_zoom[1, 1] = zoom_factor
    M_zoom[0, 2] = (frame_width - zoom_factor * frame_width) / 2
    M_zoom[1, 2] = (frame_height - zoom_factor * frame_height) / 2

    # Applica stabilizzazione e zoom frame per frame
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break # Fine del video
    
        # Logica di Trimming (Salta i frame all'inizio)
        if frame_idx < TRIM_START_FRAMES:
            frame_idx += 1
            continue
            
        # Logica di Trimming (Ferma il loop prima della fine)
        if frame_idx >= end_idx:
            break
        
        data_idx = frame_idx
        if data_idx >= len(dx_corr):
            break

        # Costruisci la matrice di trasformazione 2x3 per la correzione contenente dx, dy, dtheta
        M_stabilize = np.zeros((2, 3), dtype=np.float32)
        M_stabilize[0, 0] = np.cos(d_theta_corr[data_idx])
        M_stabilize[0, 1] = -np.sin(d_theta_corr[data_idx])
        M_stabilize[1, 0] = np.sin(d_theta_corr[data_idx])
        M_stabilize[1, 1] = np.cos(d_theta_corr[data_idx])
        M_stabilize[0, 2] = dx_corr[data_idx]
        M_stabilize[1, 2] = dy_corr[data_idx]
        
        # Applica stabilizzazione (bordi neri)
        stabilized_frame = cv2.warpAffine(frame, M_stabilize, (frame_width, frame_height), borderMode=cv2.BORDER_CONSTANT)
        
        # Applica zoom (rimuove bordi neri)
        final_frame = cv2.warpAffine(stabilized_frame, M_zoom, (frame_width, frame_height), borderMode=cv2.BORDER_CONSTANT)

        out.write(final_frame)
        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("-" * 30)
    print("Stabilizzazione + Cropping completati.")
    print(f"File salvato in: {output_video_path}")
    return True