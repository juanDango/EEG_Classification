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
    ("Proyecto", "Clasificar EEG", "Grupo")
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

elif menu == "Grupo":
    st.header("👥 Información del grupo")
    st.markdown(
        """
                **Integrantes:** Juan Daniel Castrellón  
                **Rol:** Desarrollo de back-end, diseño de modelos ML, interfaz de usuario.
        """
    )
    st.markdown("Este proyecto es parte de un trabajo académico en la Universidad de los Andes, Bogotá, Colombia")
elif menu == "Clasificar EEG":
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

        eeg_df = None
        results = None

        with st.spinner("Procesando archivo, clasificando EEG y cargando datos para visualización..."):
            archivo_parquet.seek(0)
            results = predict_eeg_from_parquet_file(archivo_parquet)

            try:
                archivo_parquet.seek(0)
                eeg_df = pd.read_parquet(archivo_parquet)
            except Exception as e:
                st.error(f"Error al leer el archivo Parquet para visualización: {e}")
                eeg_df = None

        if results and results[0] is not None and results[1] is not None:
            st.markdown("""
            <style>
                /* Estilo para pestañas inactivas */
                button[data-baseweb="tab"][aria-selected="false"] {
                    color: lightgray !important;
                    font-weight: bold !important;
                }
                /* Estilo para la pestaña activa */
                button[data-baseweb="tab"][aria-selected="true"] {
                    color: white !important;
                    font-weight: bold !important;
                }
            </style>
            """, unsafe_allow_html=True)

            tab_resultados, tab_visualizar = st.tabs(["Resultados", "Visualizar EEG"])

            with tab_resultados:
                st.subheader("Resultados de la Clasificación")
                probabilities = results[0]
                class_labels = results[1]

                df_votes = pd.DataFrame({"Votos": probabilities}, index=class_labels)
                st.bar_chart(df_votes)

                # Top 3
                top3 = df_votes.sort_values("Votos", ascending=False).head(3)
                if not top3.empty:
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
                else:
                    st.warning("No se pudieron obtener los resultados de la clasificación para el top 3.")

            with tab_visualizar:
                st.subheader("Visualización de la Señal EEG")
                if eeg_df is not None and not eeg_df.empty:
                    try:
                        channels_to_plot = eeg_df.select_dtypes(include=np.number).columns.tolist()

                        if not channels_to_plot:
                            st.warning("No se encontraron columnas numéricas (canales EEG) para graficar.")
                        else:
                            plt.style.use('dark_background')
                            fig_eeg, ax_eeg = plt.subplots(figsize=(15, 100), facecolor='black')

                            time_axis = eeg_df.index
                            if "Time" in eeg_df.columns:
                                time_axis = eeg_df["Time"]

                            ptp_s = eeg_df[channels_to_plot].apply(
                                lambda x: x.max() - x.min() if pd.notna(x.max()) and pd.notna(x.min()) else 0
                            )
                            plot_offset = 0
                            if not ptp_s.empty:
                                max_ptp = ptp_s.max()
                                if pd.notna(max_ptp) and max_ptp > 0:
                                    plot_offset = max_ptp * 1.2

                            if plot_offset == 0 and len(channels_to_plot) > 1 and eeg_df[channels_to_plot].abs().sum().sum() > 0:
                                plot_offset = 1
                            if len(channels_to_plot) <= 1:
                                plot_offset = 0

                            for i, ch_name in enumerate(channels_to_plot):
                                signal = eeg_df[ch_name].fillna(0)
                                text_x_position = (
                                    time_axis.min() - (time_axis.max() - time_axis.min()) * 0.03
                                    if not time_axis.empty and (time_axis.max() - time_axis.min()) > 0
                                    else (time_axis.min() - 0.1 if not time_axis.empty else -0.1)
                                )
                                ax_eeg.plot(time_axis, signal + i * plot_offset, color='lime', linewidth=0.8)
                                ax_eeg.text(
                                    text_x_position,
                                    i * plot_offset + signal.mean(),
                                    ch_name,
                                    va='center',
                                    ha='right',
                                    color='white',
                                    fontsize=8
                                )

                            if not time_axis.empty:
                                ax_eeg.set_xlim(time_axis.min(), time_axis.max())
                            ax_eeg.axis('off')
                            st.pyplot(fig_eeg)
                            plt.style.use('default')
                    except Exception as e:
                        st.error(f"Error al generar la gráfica del EEG: {e}")
                        plt.style.use('default')
                elif eeg_df is None:
                    st.warning("No se pudo cargar el DataFrame del EEG para la visualización debido a un error previo al leer el archivo.")
                else:
                    st.warning("El archivo Parquet está vacío o no contiene datos EEG para graficar.")
        elif results is None or (results and (results[0] is None or results[1] is None)):
            st.error("No se pudieron obtener las predicciones del modelo.")

            
            
