import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy import signal


def procesar_archivo(archivo, sampling_rate=256, duration=10, noise=0.05):
    """
    Simula una señal EEG y calcula su espectrograma.
    Retorna:
      - df:      DataFrame con columnas ['Time','EEG']
      - spec_df: DataFrame con columnas ['Time', f0, f1, ...]
    """
    # 1) Generar señal EEG sintética
    eeg_signal = nk.eeg_simulate(duration=duration,
                                 sampling_rate=sampling_rate,
                                 noise=noise)
    time = np.linspace(0, duration, sampling_rate * duration)
    df = pd.DataFrame({"Time": time, "EEG": eeg_signal})

    # 2) Calcular espectrograma (frecuencias × tiempos)
    f, t_spec, Sxx = signal.spectrogram(
        eeg_signal,
        fs=sampling_rate,
        nperseg=128,
        noverlap=64,
        scaling='density'
    )
    # Armar DataFrame de espectrograma: filas=instantes de tiempo, columnas=frecuencias
    spec_df = pd.DataFrame(Sxx.T, columns=np.round(f, 2))
    spec_df.insert(0, "Time", t_spec)

    return df, spec_df


def run_model(data):
    """Simula un modelo de clasificación con 6 salidas."""
    simulated_model = np.random.rand(6)
    return simulated_model / simulated_model.sum()