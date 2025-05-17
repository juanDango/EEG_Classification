# Clasificador de EEG

![Streamlit](https://img.shields.io/badge/Streamlit-v1.0-orange) ![Python](https://img.shields.io/badge/Python-3.9-blue) ![Docker](https://img.shields.io/badge/Docker-ready-blue)

Una aplicación web interactiva construida con Streamlit para clasificar señales de EEG en seis categorías: `Seizure`, `LPD`, `GPD`, `LRDA`, `GRDA` y `Otros`.

## ⚙️ Requisitos previos

* Python 3.8 o superior
* Docker

---

## 📥 Instalación local

1. Clona el repositorio:

   ```bash
   git clone https://github.com/tu_usuario/eeg-classifier.git
   cd eeg-classifier
   ```
2. Crea y activa un entorno virtual:

   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate    # Windows
   ```
3. Instala las dependencias:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. Ejecuta la app de Streamlit:

   ```bash
   streamlit run app.py
   ```
5. Abre tu navegador en `http://localhost:8501`.

---

## 🐳 Uso con Docker

1. Construye la imagen:

   ```bash
   docker build -t eeg-classifier .
   ```
2. Ejecuta el contenedor (puerto 8600 como ejemplo):

   ```bash
   docker run -p 8600:8501 eeg-classifier
   ```
3. Accede en `http://localhost:8600`.

> Si el puerto `8501` está ocupado, ajusta el mapeo `-p <PUERTO_HOST>:8501`.

---

## 🛠 Estructura del proyecto

```text
├── .streamlit/
│   └── config.toml       # Configuración de tema (dark)
├── Dockerfile            # Definición de la imagen Docker
├── requirements.txt      # Librerías Python
├── app.py                # Aplicación Streamlit principal
├── logic.py              # Funciones de procesamiento y modelo
├── utils/
│   └── patologias.py     # Diccionario de patologías y metadatos
└── README.md             # Documentación del proyecto
```

---

## 🔧 Personalización

* **Simulación EEG**: ajusta `sampling_rate`, `duration`, `noise` o número de canales en `procesar_archivo`.
* **Modelo ML**: implementa tu función `run_model(data)` en `logic.py`.
* **Tema y estilo**: modifica `.streamlit/config.toml` para colores y fuentes.

---

## 👥 Equipo

* **Juan Daniel Castrellón** – backend, lógica de simulación y despliegue.

---

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.
