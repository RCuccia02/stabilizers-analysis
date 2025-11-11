import cv2
import numpy as np
import os

def run_phase1(video_file_path, output_dir, video_name_base):
    """
    Esegue la Fase 1: Estrazione Feature e Calcolo Traiettoria.
    Salva sia la traiettoria accumulata (X_act) che i vettori (V_act).
    
    Ritorna i percorsi ai due file di dati.
    """
    print(f"--- Avvio Fase 1: Estrazione Feature per {video_file_path} ---")
    
    output_video_file = os.path.join(output_dir, f"{video_name_base}_with_points.mp4")
    output_data_X_act = os.path.join(output_dir, "traiettoria_rumorosa_X_act.npy")
    output_data_V_act = os.path.join(output_dir, "vettori_rumorosi_V_act.npy")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        print(f"ERRORE: Impossibile aprire {video_file_path}")
        return None, None

    # Setup Video Writer
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))

    # Leggi il primo frame
    ret, prev_frame = cap.read()
    if not ret:
        print("ERRORE: Impossibile leggere il primo frame.")
        cap.release()
        out.release() 
        return None, None

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Rileva punti di interesse nel primo frame
    MAX_PUNTI = 200 
    prev_points = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=MAX_PUNTI, qualityLevel=0.1,
        minDistance=7, blockSize=7
    )
    if prev_points is None:
        print("ERRORE: Nessun punto trovato nel primo frame.")
        cap.release()
        out.release() 
        return None, None

    print(f"Trovati {len(prev_points)} punti iniziali da tracciare.")

    trajectory_X_act = [(0.0, 0.0, 0.0)] # Traiettoria accumulata
    vectors_V_act = [(0.0, 0.0, 0.0)]    # Vettori (V_act(n))
    frame_count = 0

    # Ciclo sui frame del video
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            print("Fine del video.")
            break 
        
        frame_count += 1
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        dx, dy, d_theta = 0.0, 0.0, 0.0
        frame_with_points = curr_frame.copy()

        # Calcola il flusso ottico
        curr_points, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_points, None
        )

        if curr_points is not None:
            good_new = curr_points[status == 1]
        else:
            good_new = np.array([])

        if len(good_new) < 50:
            # TRACCIAMENTO FALLITO - si cercano nuovi punti
            print(f"Attenzione: Tracciamento fallito al frame {frame_count}. Riavvio dei punti.")
            new_points = cv2.goodFeaturesToTrack(
                curr_gray, maxCorners=MAX_PUNTI, qualityLevel=0.1,
                minDistance=7, blockSize=7
            )
            prev_points = new_points
            
            if new_points is not None:
                for point in new_points:
                    x, y = point.ravel()
                    cv2.drawMarker(frame_with_points, (int(x), int(y)), 
                                   color=(0, 0, 255), markerType=cv2.MARKER_CROSS, 
                                   markerSize=5, thickness=1)
        else:
            # TRACCIAMENTO RIUSCITO
            good_old = prev_points[status == 1]
            for point in good_new:
                x, y = point.ravel()
                cv2.drawMarker(frame_with_points, (int(x), int(y)), 
                               color=(0, 255, 0), markerType=cv2.MARKER_CROSS, 
                               markerSize=5, thickness=1)
            
            # Stima la trasformazione affine tra i punti vecchi e nuovi
            m, _ = cv2.estimateAffinePartial2D(good_old, good_new, ransacReprojThreshold=3)
            if m is not None:
                dx = m[0, 2]
                dy = m[1, 2]
                d_theta = np.arctan2(m[1, 0], m[0, 0])
            
            prev_points = good_new.reshape(-1, 1, 2)
        
        out.write(frame_with_points)
        
        # Aggiorna traiettoria e vettori
        vectors_V_act.append((dx, dy, d_theta))
        
        last_x, last_y, last_theta = trajectory_X_act[-1]
        new_theta = last_theta + d_theta
        new_x = last_x + (dx * np.cos(last_theta) - dy * np.sin(last_theta))
        new_y = last_y + (dx * np.sin(last_theta) + dy * np.cos(last_theta))
        trajectory_X_act.append((new_x, new_y, new_theta))
        
        prev_gray = curr_gray.copy()

    print(f"Fase 1 completata. Processati {frame_count} frame.")
    
    trajectory_array = np.array(trajectory_X_act)
    vectors_array = np.array(vectors_V_act)
    
    np.save(output_data_X_act, trajectory_array)
    np.save(output_data_V_act, vectors_array)

    print(f"Salvati {trajectory_array.shape} dati in: {output_data_X_act}")
    print(f"Salvati {vectors_array.shape} dati in: {output_data_V_act}")
    
    cap.release()
    out.release() 
    cv2.destroyAllWindows()
    
    return output_data_X_act, output_data_V_act