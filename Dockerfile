# Usa una imagen oficial de Python ligera
FROM python:3.11.9-slim

# Establece el directorio de trabajo
WORKDIR /app

# Copia el archivo de dependencias y instala
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto del código de la aplicación
COPY . .

# Expone el puerto por defecto de Streamlit
EXPOSE 8501

# Variables de entorno para que Streamlit escuche en todas las interfaces
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501

# Comando para arrancar la app
ENTRYPOINT ["streamlit", "run", "app.py"]
