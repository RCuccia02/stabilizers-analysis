import argparse
import os
import sys

# 1. IMPORTAZIONE DEI MODULI NECESSARI
try:
    import phase1_extract
    import phase2_filters
    import phase3_stabilize
except ImportError as e:
    print(f"ERRORE: Impossibile importare i moduli: {e}")
    print("Assicurati che 'phase1_extract.py', 'phase2_filters.py', e 'phase3_stabilize.py' siano nella stessa cartella.")
    sys.exit(1)

# 2. DEFINIZIONE DELLA FUNZIONE MAIN E DELLE COSTANTI
BASE_INPUT_DIR = "./inputs"
BASE_OUTPUT_DIR = "./outputs"

def main(video_path, algorithm, smoothing_method):
    """
    Orchestra l'intera pipeline di stabilizzazione.
    """
    print("--- AVVIO PIPELINE ---")
    print(f"  Video Sorgente: {video_path}")
    print(f"  Algoritmo Selezionato: {algorithm}")
    print(f"  Metodo di Smoothing: {smoothing_method}")
    print("-" * 30)

    # Verifica che il file video esista
    if not os.path.exists(video_path):
        print(f"ERRORE CRITICO: File video non trovato: {video_path}")
        sys.exit(1)
        
    # definizione del nome base del video e delle cartelle di output
    video_name_base = os.path.splitext(os.path.basename(video_path))[0]

    phase1_output_dir = os.path.join(BASE_OUTPUT_DIR, "phase1", video_name_base)
    
    
    # Fase 1: Estrazione Feature e Calcolo Traiettoria
    x_act_path, v_act_path = phase1_extract.run_phase1(
        video_file_path=video_path,
        output_dir=phase1_output_dir,
        video_name_base=video_name_base
    )
    
    if x_act_path is None:
        print("ERRORE CRITICO: Fase 1 (Estrazione Feature) fallita. Interruzione.")
        sys.exit(1)

    print("-" * 30)

    # Fase 2: Filtraggio della Traiettoria
    x_smooth_path = None
    trim_config = {} # Il trimming è specifico per FPS
    phase2_output_dir = os.path.join(BASE_OUTPUT_DIR, f"phase2_{algorithm}", video_name_base)

    if algorithm == "FPS":
        x_smooth_path = phase2_filters.run_fps_filter(
            x_act_path=x_act_path,
            output_dir=phase2_output_dir,
            video_name=video_name_base,
            smoothing_method=smoothing_method,
            cutoff=0.03, # Non usato nel gaussiano
            sigma=0.02 # Il nostro valore ottimizzato
        )
        # FPS richiede trimming per evitare uno zoom eccessivo
        trim_config = {"start": 0, "end": 0}
        if trim_config["start"] != 0 or trim_config["end"] != 0:
            print(f"Trimming ({trim_config['start']} / {trim_config['end']} frame) abilitato per FPS.")

    elif algorithm == "MVI":
        x_smooth_path = phase2_filters.run_mvi_filter(
            v_act_path=v_act_path,
            x_act_path=x_act_path, 
            output_dir=phase2_output_dir,
            video_name=video_name_base,
            delta=0.90 # Damping factor
        )
        # MVI è real-time, non richiede trimming
        trim_config = {}

    elif algorithm == "Kalman":
        x_smooth_path = phase2_filters.run_kalman_filter(
            x_act_path=x_act_path,
            output_dir=phase2_output_dir,
            video_name=video_name_base,
            R_val=20.0, # Rumore dei dati
            Q_val=0.001 # Rumore del modello
        )
        # Kalman è real-time, non richiede trimming
        trim_config = {}
        
    elif algorithm == "DL":
        print("ERRORE: Filtro DL non ancora implementato.")
        sys.exit(1)

    else:
        print(f"ERRORE: Algoritmo '{algorithm}' non riconosciuto.")
        sys.exit(1)

    if x_smooth_path is None:
        print(f"ERRORE CRITICO: Fase 2 ({algorithm}) fallita. Interruzione.")
        sys.exit(1)

    print("-" * 30)

    # Fase 3: Stabilizzazione Video
    phase3_output_dir = os.path.join(BASE_OUTPUT_DIR, "phase3_final_videos")
    final_video_path = os.path.join(phase3_output_dir, f"{video_name_base}_stabilizzato_{algorithm}_{smoothing_method}.mp4")
    
    success = phase3_stabilize.run_phase3(
        video_input_path=video_path,
        x_act_path=x_act_path,
        x_smooth_path=x_smooth_path,
        output_video_path=final_video_path,
        trim_config=trim_config
    )

    if not success:
        print("ERRORE CRITICO: Fase 3 (Stabilizzazione) fallita. Interruzione.")
        sys.exit(1)
        
    print("-" * 30)
    print("--- PIPELINE COMPLETATA CON SUCCESSO ---")
    print(f"Video finale salvato in: {final_video_path}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Pipeline di Stabilizzazione Video")
    
    parser.add_argument(
        "video_path", 
        type=str, 
        help="Il percorso del file video da stabilizzare (es. ./inputs/video.mp4)"
    )
    parser.add_argument(
        "algorithm", 
        type=str, 
        choices=["FPS", "MVI", "Kalman", "DL"], 
        help="L'algoritmo di filtraggio da utilizzare"
    )

    parser.add_argument(
        "--smoothing_method", "-s",
        type=str, 
        choices=["gaussian", "cutoff"], 
        help="Il metodo di smoothing da utilizzare (solo per FPS)",
        required=False, default=""
    )
    args = parser.parse_args()
    
    main(args.video_path, args.algorithm, args.smoothing_method)