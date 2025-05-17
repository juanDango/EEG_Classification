import streamlit as st
import pandas as pd
from logic import procesar_archivo, run_model
from utils.patologias import patologias
import matplotlib.pyplot as plt

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
    st.markdown("""
        "- **Procesamiento de señales:** Extracción de características del EEG mediante transformadas y filtros.  
- **Modelo:** Ensamble de redes neuronales convolucionales con capa BiLSTM y salida Softmax.  
- **Interfaz:** Aplicación web con Streamlit para cargar archivos y mostrar resultados."""
    )

elif menu == "Clasificar EEG":
    st.subheader("Sube un archivo de EEG para clasificarlo")
    archivo = st.file_uploader("Selecciona un archivo", type=None)

    if archivo is not None:
        st.success("Archivo recibido")

        # Dentro de tu sección de Streamlit tras procesar archivo:
        df = procesar_archivo(archivo)  # DataFrame con columnas 'Time' y 'EEG'

        # 1. Aplicar estilo oscuro global
        plt.style.use('dark_background')

        # 2. Crear figura y ejes
        fig, ax = plt.subplots(figsize=(10, 4), facecolor='black')

        # 3. Graficar señal sintética o real
        ax.plot(df["Time"], df["EEG"], color='lime', linewidth=1)

        # 4. Ajustar márgenes y ocultar ejes
        ax.margins(x=0, y=0.05)
        ax.set_axis_off()

        # 5. Mostrar en Streamlit
        st.pyplot(fig)
        resultado = run_model(df)

        etiquetas = ["Seizure", "LPD", "GPD", "LRDA", "GRDA", "Otros"]
        df = pd.DataFrame({"Votos": resultado}, index=etiquetas)

        # Gráfico de barras
        st.subheader("Resultados de la clasificación")
        st.bar_chart(df)

        # Top 3
        top3 = df.sort_values("Votos", ascending=False).head(3)
        # Etiqueta más probable grande
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
    st.markdown("""
            **Integrantes:** Juan Daniel Castrellón 
            **Rol:** Desarrollo de back-end, diseño de modelos ML, interfaz de usuario."""
        )
    st.markdown("Este proyecto es parte de un trabajo académico en la Universidad de los Andes, Bogotá, Colombia")