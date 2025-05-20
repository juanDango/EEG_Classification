# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from logic import procesar_archivo, run_model, predict_eeg_from_parquet_file,load_eeg_classification_model
from utils.patologias import patologias
import io 
from tensorflow.keras.utils import plot_model
# Título principal
st.set_page_config(page_title="Clasificador de EEG", layout="wide")
st.title("Clasificador de EEG")

# Sidebar con navegación
menu = st.sidebar.radio(
    "Navegación",
    ("Proyecto", "Clasificar EEG", "Grupo", 'Clasificar eegs 2')
)

if menu == "Proyecto":
    st.header("📝 Información del proyecto")
    st.markdown(
        "Este proyecto tiene como objetivo desarrollar un clasificador de patrones de EEG para seis categorías: Seizure, LPD, GPD, LRDA, GRDA y Otros."
    )
    st.markdown(
        """
        - **Procesamiento de señales:** Extracción de características del EEG mediante transformadas y filtros.  
        - **Modelo:** Ensamble de redes neuronales convolucionales con capa BiLSTM y salida Softmax.  
        - **Interfaz:** Aplicación web con Streamlit para cargar archivos y mostrar resultados.
        """
    )

elif menu == "Clasificar EEG":
    st.subheader("Sube un archivo de EEG para clasificarlo")
    archivo = st.file_uploader("Selecciona un archivo", type=None)

    if archivo is not None:
        st.success("Archivo recibido")

        # Indicador de carga con spinner y barra de progreso
        with st.spinner("Procesando archivo y clasificando EEG..."):
            progress = st.progress(0)

            # 1. Simular señal EEG y extraer espectrograma
            df, spec_df = procesar_archivo(archivo)
            progress.progress(20)

            # 2. Mostrar espectrograma
            fig_spec, ax_spec = plt.subplots(figsize=(10, 3))
            times = spec_df["Time"].values
            spec_data = spec_df.drop("Time", axis=1).T.values
            freq_labels = spec_df.drop("Time", axis=1).columns
            ax_spec.imshow(
                spec_data,
                aspect="auto",
                origin="lower",
                extent=[times[0], times[-1], 0, len(freq_labels)]
            )
            ax_spec.set_yticks(np.arange(len(freq_labels)) + 0.5)
            ax_spec.set_yticklabels(freq_labels, fontsize=6)
            ax_spec.set_xlabel("Tiempo (s)")
            ax_spec.set_ylabel("Frecuencia (Hz)")
            ax_spec.set_title("Espectrograma sintético")
            st.pyplot(fig_spec)
            progress.progress(40)

            # 3. Visualización estilo clínico
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(10, 4), facecolor='black')
            channels = df.columns.drop("Time")
            ptp_vals = df[channels].apply(lambda x: x.max() - x.min())
            offset = ptp_vals.max() * 1.2
            for i, col in enumerate(channels):
                ax.plot(
                    df["Time"],
                    df[col] + i * offset,
                    color='lime', linewidth=0.8
                )
                ax.text(
                    -0.01 * df["Time"].max(),
                    i * offset,
                    col,
                    va='center', ha='right',
                    color='white', fontsize=8
                )
            ax.set_xlim(df["Time"].min(), df["Time"].max())
            ax.axis('off')
            st.pyplot(fig)
            progress.progress(60)

            # 4. Clasificación con el modelo
            resultado = run_model(df)
            progress.progress(100)
            del progress

        st.success("Procesamiento completado!")

        etiquetas = ["Seizure", "LPD", "GPD", "LRDA", "GRDA", "Otros"]
        df_votes = pd.DataFrame({"Votos": resultado}, index=etiquetas)

        # Gráfico de barras
        st.subheader("Resultados de la clasificación")
        st.bar_chart(df_votes)

        # Top 3
        top3 = df_votes.sort_values("Votos", ascending=False).head(3)
        mejor = top3.index[0]
        pat_mejor = patologias.get(mejor)
        if pat_mejor:
            st.markdown(f"# 1. {pat_mejor['nombre_completo']}")
            st.write(f"**ID:** {pat_mejor['id']}")
            st.write(f"**Probabilidad:** {top3.loc[mejor, 'Votos']:.2f}")
            st.markdown("**Descripción:**")
            st.markdown(pat_mejor["descripción"], unsafe_allow_html=True)
            st.markdown("**Posibles tratamientos:**")
            st.markdown(pat_mejor["posibles_tratamientos"], unsafe_allow_html=True)
            if pat_mejor.get("citas"):
                st.markdown("**Citas / Referencias:**")
                for cita in pat_mejor["citas"]:
                    st.markdown(f"- [{cita}]({cita})")
            st.markdown("---")

        # Acordeones para 2 y 3
        for idx in top3.index[1:]:
            pat = patologias.get(idx)
            if pat:
                with st.expander(f"{pat['nombre_completo']} ({idx})"):
                    st.write(f"**ID:** {pat['id']}")
                    st.write(f"**Probabilidad:** {top3.loc[idx, 'Votos']:.2f}")
                    st.markdown("**Descripción:**")
                    st.markdown(pat["descripción"], unsafe_allow_html=True)
                    st.markdown("**Posibles tratamientos:**")
                    st.markdown(pat["posibles_tratamientos"], unsafe_allow_html=True)
                    if pat.get("citas"):
                        st.markdown("**Citas / Referencias:**")
                        for cita in pat["citas"]:
                            st.markdown(f"- [{cita}]({cita})")
            else:
                st.warning(f"No se encontró información para: {idx}")

elif menu == "Grupo":
    st.header("👥 Información del grupo")
    st.markdown(
        """
                **Integrantes:** Juan Daniel Castrellón  
                **Rol:** Desarrollo de back-end, diseño de modelos ML, interfaz de usuario.
        """
    )
    st.markdown("Este proyecto es parte de un trabajo académico en la Universidad de los Andes, Bogotá, Colombia")
elif menu == "Clasificar eegs 2":
    #st.markdown("#### Resumen del Modelo")
    #model = load_eeg_classification_model()
    #summary_string = io.StringIO()
    #model.summary(print_fn=lambda x: summary_string.write(x + '\n'))
    #st.text(summary_string.getvalue())

    # 2. Mostrar Gráfico de la Arquitectura del Modelo (Keras)
    #st.markdown("#### Gráfico de la Arquitectura")
    #try:
        # Guardar la imagen temporalmente y mostrarla
        #plot_path = "model_plot.png"
        #plot_model(model, to_file=plot_path, show_shapes=True, show_layer_names=True, rankdir='TB') # TB: Top to Bottom, LR: Left to Right
        #st.image(plot_path)
    #except Exception as e:
        #st.warning(f"No se pudo generar la gráfica del modelo. Asegúrate de tener Graphviz y Pydot instalados y configurados en tu sistema.")
        #st.warning(f"Error: {e}")
    st.subheader("Sube un archivo de EEG para clasificarlo")
    archivo_parquet = st.file_uploader("Selecciona un archivo Parquet", type=["parquet"])

    if archivo_parquet is not None:
        st.success("Archivo recibido")
        st.subheader("Resultados de Clasificación")
        with st.spinner("Procesando archivo y clasificando EEG..."):

            archivo_parquet.seek(0) 
            results = predict_eeg_from_parquet_file(archivo_parquet)
            df_results = pd.DataFrame(
                [results[0]],
                columns=results[1]
            )
            st.table(df_results)
        # --- Display EEG Image ---
        st.subheader("Visualización de la Señal EEG")
        try:

            archivo_parquet.seek(0) 
            eeg_df = pd.read_parquet(archivo_parquet)

            eeg_channel_columns = eeg_df.select_dtypes(include=np.number).columns.tolist()
            
            if not eeg_channel_columns:
                st.warning("No se encontraron columnas numéricas para graficar como canales EEG.")
            elif len(eeg_df) == 0:
                st.warning("El archivo Parquet está vacío o no contiene datos para graficar.")
            else:
                # 3. Plot the signals
                fig, axes = plt.subplots(len(eeg_channel_columns), 1, figsize=(12, 2 * len(eeg_channel_columns)), sharex=True)
                if len(eeg_channel_columns) == 1: 
                    axes = [axes] 

                for i, channel in enumerate(eeg_channel_columns):
                    axes[i].plot(eeg_df.index, eeg_df[channel]) 
                    axes[i].set_ylabel(channel)
                    axes[i].grid(True)
                
                axes[-1].set_xlabel("Muestras (o Tiempo)")
                fig.suptitle("Señales EEG", fontsize=16)
                plt.tight_layout(rect=[0, 0, 1, 0.96]) 

                st.pyplot(fig)

        except Exception as e:
            st.error(f"Error al procesar o graficar el archivo EEG: {e}")
        
        




