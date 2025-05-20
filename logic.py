import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy import signal

import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
import tensorflow as tf
import streamlit as st
import os

import tensorflow as tf
from tensorflow.keras.utils import plot_model # Para graficar el modelo
import io

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

class CFG:
    verbose = 5
    seed = 77
    image_size = [400, 300]
    epochs = 13
    batch_size = 32
    lr_mode = "cos"
    drop_remainder = True

    num_classes = 6
    class_names = ["Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"]
    label2name = dict(enumerate(class_names))
    name2label = {v: k for k, v in label2name.items()}
    N_TRIALS_OPTUNA = 30
    FS = 200
    CLIP_MIN = -1024
    CLIP_MAX = 1024
    FILTER_LOWCUT = 0.5
    FILTER_HIGHCUT = 40
    FILTER_ORDER = 4
    EXPECTED_EEG_CHANNELS = ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1',
                             'Fz', 'Cz', 'Pz',
                             'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']
    BIPOLAR_MONTAGE_DEFINITIONS = {
        'LL': ['Fp1', 'F7', 'T3', 'T5', 'O1'],
        'RL': ['Fp2', 'F8', 'T4', 'T6', 'O2'],
        'LP': ['Fp1', 'F3', 'C3', 'P3', 'O1'],
        'RP': ['Fp2', 'F4', 'C4', 'P4', 'O2']
    }

    SEGMENT_DURATION_SEC = 10
    CNN_WINDOW_DURATION_SEC = 2
    CNN_WINDOW_STRIDE_SEC = 1

    CNN_OUTPUT_FEATURES = 128
    BILSTM_UNITS = 64

CFG.BRAIN_LEAD_CHAINS = [CFG.BIPOLAR_MONTAGE_DEFINITIONS[key] for key in CFG.BIPOLAR_MONTAGE_DEFINITIONS]
CFG.N_BIPOLAR_DERIVATIONS = sum(len(chain) - 1 for chain in CFG.BRAIN_LEAD_CHAINS)

CFG.SAMPLES_PER_SEGMENT = CFG.FS * CFG.SEGMENT_DURATION_SEC
CFG.CNN_WINDOW_SAMPLES = CFG.FS * CFG.CNN_WINDOW_DURATION_SEC
CFG.CNN_STRIDE_SAMPLES = CFG.FS * CFG.CNN_WINDOW_STRIDE_SEC
CFG.NUM_CNN_WINDOWS_PER_SEGMENT = (CFG.SAMPLES_PER_SEGMENT - CFG.CNN_WINDOW_SAMPLES) // CFG.CNN_STRIDE_SAMPLES + 1
FS = CFG.FS
ETIQUETAS_CLASES = ["Seizure", "LPD", "GPD", "LRDA", "GRDA", "Otros"]
CLIP_MIN = CFG.CLIP_MIN
CLIP_MAX = CFG.CLIP_MAX
FILTER_LOWCUT = CFG.FILTER_LOWCUT
FILTER_HIGHCUT = CFG.FILTER_HIGHCUT
FILTER_ORDER = CFG.FILTER_ORDER
EXPECTED_EEG_CHANNELS = CFG.EXPECTED_EEG_CHANNELS
BRAIN_LEAD_CHAINS = CFG.BRAIN_LEAD_CHAINS
MODEL_PATH = './final_eeg_model.keras'
verbose = 5
seed = 77
image_size = [400, 300]
epochs = 13
batch_size = 32
lr_mode = "cos"
drop_remainder = True

num_classes = 6
class_names = ["Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"]
label2name = dict(enumerate(class_names))
name2label = {v: k for k, v in label2name.items()}
N_TRIALS_OPTUNA = 30
FS = 200
CLIP_MIN = -1024
CLIP_MAX = 1024
FILTER_LOWCUT = 0.5
FILTER_HIGHCUT = 40
FILTER_ORDER = 4
EXPECTED_EEG_CHANNELS = ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1',
                        'Fz', 'Cz', 'Pz',
                        'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']
BIPOLAR_MONTAGE_DEFINITIONS = {
'LL': ['Fp1', 'F7', 'T3', 'T5', 'O1'],
'RL': ['Fp2', 'F8', 'T4', 'T6', 'O2'],
'LP': ['Fp1', 'F3', 'C3', 'P3', 'O1'],
'RP': ['Fp2', 'F4', 'C4', 'P4', 'O2']
}

SEGMENT_DURATION_SEC = 10
CNN_WINDOW_DURATION_SEC = 2
CNN_WINDOW_STRIDE_SEC = 1

CNN_OUTPUT_FEATURES = 128
BILSTM_UNITS = 64


def calculate_bipolar_derivations(data_transposed, channel_names_list, montage_chains):
    leads_map = {name: i for i, name in enumerate(channel_names_list)}
    derived_signals = []
    for chain in montage_chains:
        for i in range(len(chain) - 1):
            ch1_name, ch2_name = chain[i], chain[i+1]
            if ch1_name in leads_map and ch2_name in leads_map:
                ch1_idx, ch2_idx = leads_map[ch1_name], leads_map[ch2_name]
                derived_signals.append(data_transposed[ch1_idx, :] - data_transposed[ch2_idx, :])
            else:
                if CFG.verbose > 0:
                    print(f"Advertencia: Canales {ch1_name} o {ch2_name} no encontrados para la derivación. Saltando.")
    if not derived_signals:
        return np.array([]).reshape(0, data_transposed.shape[1] if data_transposed.ndim > 1 else 0)
    return np.array(derived_signals)

def butter_bandpass_filter(data, lowcut, highcut, fs, order=FILTER_ORDER):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = None, None 
    valid_bandpass = True
    if high >= 1.0: high = 0.999
    if low <= 0.0: low = 1e-6 
    if low >= high: 
        valid_bandpass = False
        if lowcut == 0 and highcut > 0 and highcut < fs/2 : 
             b, a = butter(order, high, btype='lowpass', analog=False)
        elif highcut >= fs/2 and lowcut > 0 and lowcut < fs/2: 
             b, a = butter(order, low, btype='highpass', analog=False)
        else:
            if CFG.verbose > 0:
                print(f"Error en los parámetros del filtro: lowcut={lowcut}, highcut={highcut}, fs={fs}. No se puede diseñar el filtro.")
            return data 
    
    if valid_bandpass and low < high : 
        b, a = butter(order, [low, high], btype='band', analog=False)
    elif not (b is not None and a is not None): 
        if CFG.verbose > 0:
            print(f"Advertencia: No se pudo diseñar un filtro adecuado con lowcut={lowcut}, highcut={highcut}. Devolviendo datos sin filtrar.")
        return data

    y = lfilter(b, a, data, axis=-1)
    return y

def load_eeg_classification_model():
    """Carga el modelo Keras pre-entrenado."""
    try:
        if CFG.verbose > 0:
            print(f"Intentando cargar modelo desde: {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)
        if CFG.verbose > 0:
            print("Modelo EEG cargado exitosamente.")
        return model
    except Exception as e:
        print(f"Error crítico al cargar el modelo EEG desde {MODEL_PATH}: {e}")
        return None
def preprocess_parquet_for_model(parquet_path_or_buffer):
    """
    Lee un archivo Parquet (desde ruta o buffer), lo procesa y devuelve
    los datos listos para la entrada del modelo Keras.
    """
    try:

        eeg_df = pd.read_parquet(parquet_path_or_buffer)
    except Exception as e:
        if CFG.verbose > 0:
            print(f"Error al leer el archivo Parquet '{parquet_path_or_buffer}': {e}")
        return None


    available_channels = [ch for ch in EXPECTED_EEG_CHANNELS if ch in eeg_df.columns]
    if not available_channels:
        if CFG.verbose > 0:
            print("Error: Ninguno de los canales EEG esperados se encontró en el archivo.")
        return None
    
    eeg_segment_df = eeg_df[available_channels].copy() 

    if len(eeg_segment_df) < CFG.SAMPLES_PER_SEGMENT:
        if CFG.verbose > 0:
            print(f"Advertencia: Segmento EEG corto ({len(eeg_segment_df)} de {CFG.SAMPLES_PER_SEGMENT}). Rellenando con ceros.")
        padding_length = CFG.SAMPLES_PER_SEGMENT - len(eeg_segment_df)
        padding = pd.DataFrame(np.zeros((padding_length, len(eeg_segment_df.columns))), columns=eeg_segment_df.columns)
        eeg_segment_df = pd.concat([eeg_segment_df, padding], ignore_index=True)
    elif len(eeg_segment_df) > CFG.SAMPLES_PER_SEGMENT:
        if CFG.verbose > 0:
            print(f"Advertencia: Segmento EEG largo ({len(eeg_segment_df)} de {CFG.SAMPLES_PER_SEGMENT}). Truncando.")
        eeg_segment_df = eeg_segment_df.iloc[:CFG.SAMPLES_PER_SEGMENT]

    waves_data = eeg_segment_df.values.astype(np.float64)


    if np.isnan(waves_data).any():
        if CFG.verbose > 0:
            print("Advertencia: NaNs encontrados. Imputando con media del canal.")
        col_means = np.nanmean(waves_data, axis=0)
        col_means = np.nan_to_num(col_means, nan=0.0)
        inds = np.where(np.isnan(waves_data))
        waves_data[inds] = np.take(col_means, inds[1])
        waves_data = np.nan_to_num(waves_data, nan=0.0)

    waves_transposed = waves_data.T 

    # Derivaciones bipolares
    bipolar_waves = calculate_bipolar_derivations(waves_transposed, available_channels, BRAIN_LEAD_CHAINS)

    if bipolar_waves.shape[0] == 0:
        if CFG.verbose > 0:
            print("Error: No se pudieron generar derivaciones bipolares.")
        return None

    print(bipolar_waves.shape)
    if bipolar_waves.shape[0] != CFG.N_BIPOLAR_DERIVATIONS:
        if CFG.verbose > 0:
            print(f"Advertencia: Se generaron {bipolar_waves.shape[0]} derivaciones bipolares en lugar de {CFG.N_BIPOLAR_DERIVATIONS}. Ajustando tamaño.")
        adjusted_bipolar_waves = np.zeros((CFG.N_BIPOLAR_DERIVATIONS, bipolar_waves.shape[1]))

        num_derivs_to_copy = min(bipolar_waves.shape[0], CFG.N_BIPOLAR_DERIVATIONS)
        adjusted_bipolar_waves[:num_derivs_to_copy, :] = bipolar_waves[:num_derivs_to_copy, :]
        bipolar_waves = adjusted_bipolar_waves
        
    # Clipping y Filtrado
    clipped_waves = np.clip(bipolar_waves, CLIP_MIN, CLIP_MAX)
    filtered_waves = butter_bandpass_filter(clipped_waves, FILTER_LOWCUT, FILTER_HIGHCUT, FS, order=FILTER_ORDER)



    model_input = filtered_waves 
    #model_input = np.expand_dims(model_input, axis=0) 
    

    
    if CFG.verbose > 0:
        print(f"Forma final de entrada al modelo: {model_input.shape}")
    return model_input
def calculate_bipolar_derivations(data_transposed, channel_names_list, montage_chains):
    leads_map = {name: i for i, name in enumerate(channel_names_list)}
    derived_signals = []
    for chain in montage_chains:
        for i in range(len(chain) - 1):
            ch1_name, ch2_name = chain[i], chain[i+1]
            if ch1_name in leads_map and ch2_name in leads_map:
                ch1_idx, ch2_idx = leads_map[ch1_name], leads_map[ch2_name]
                derived_signals.append(data_transposed[ch1_idx, :] - data_transposed[ch2_idx, :])
            else:
                if CFG.verbose > 0:
                    print(f"Advertencia: Canales {ch1_name} o {ch2_name} no encontrados para la derivación. Saltando.")
    if not derived_signals:
        return np.array([]).reshape(0, data_transposed.shape[1] if data_transposed.ndim > 1 else 0)
    return np.array(derived_signals)

def create_cnn_patches(eeg_segment_data: np.ndarray):
    """
    Creates overlapping patches from a 10-second EEG segment.
    Args:
        eeg_segment_data (np.array): Processed EEG data (N_DERIVATIONS, SAMPLES_PER_SEGMENT).
    Returns:
        np.array: Patches (NUM_CNN_WINDOWS_PER_SEGMENT, N_DERIVATIONS, CNN_WINDOW_SAMPLES).
                  Returns None if input data is not as expected.
    """
    print(eeg_segment_data.shape)
    if eeg_segment_data is None or eeg_segment_data.shape[1] != CFG.SAMPLES_PER_SEGMENT:
        if CFG.verbose > 0 and eeg_segment_data is not None:
             print(f"Error en create_cnn_patches: Se esperaban {CFG.SAMPLES_PER_SEGMENT} muestras, se obtuvieron {eeg_segment_data.shape[1]}")
        return None
    if eeg_segment_data.shape[0] != CFG.N_BIPOLAR_DERIVATIONS:
        if CFG.verbose > 0:
            print(f"Error en create_cnn_patches: Se esperaban {CFG.N_BIPOLAR_DERIVATIONS} derivaciones, se obtuvieron {eeg_segment_data.shape[0]}")
        return None


    patches = []
    for i in range(CFG.NUM_CNN_WINDOWS_PER_SEGMENT):
        start = i * CFG.CNN_STRIDE_SAMPLES
        end = start + CFG.CNN_WINDOW_SAMPLES
        patch = eeg_segment_data[:, start:end]
        patches.append(patch)

    return np.array(patches)
def predict_eeg_from_parquet_file(parquet_file_buffer):
    """
    Carga el modelo, preprocesa los datos del archivo Parquet (buffer) y devuelve las predicciones.
    """
    model = load_eeg_classification_model()
    if model is None:
        #st.error("El modelo de clasificación no pudo ser cargado. No se puede predecir.")
        return None, None 

    model_input = preprocess_parquet_for_model(parquet_file_buffer)
    if model_input is None:
        #st.error("Los datos del EEG no pudieron ser procesados. No se puede predecir.")
        return None, None
    else: 
        model_changed= create_cnn_patches(model_input)
        model_changed=np.expand_dims(model_changed, axis=0)
    try:
        raw_predictions = model.predict(model_changed)
        probabilities = raw_predictions[0]
        return probabilities, ETIQUETAS_CLASES
    except Exception as e:
        print(f"Error durante la predicción del modelo: {e}")
        #st.error(f"Ocurrió un error durante la predicción: {e}")
        return None, None

