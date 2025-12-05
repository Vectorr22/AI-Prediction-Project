# 🔮 El Oráculo de la Lotería Mexicana (AI Powered)

![Banner Principal](screenshots/banner.png)


Un sistema interactivo de **Visión por Computadora** e **Inteligencia Artificial Generativa** que moderniza la tradición de la Lotería Mexicana.

Esta aplicación es capaz de reconocer las cartas del juego en tiempo real utilizando una cámara web y generar "predicciones místicas" personalizadas, narradas con voz dramática, conectando el pasado, presente y futuro del usuario basándose en las cartas detectadas.

---

## 📸 Demo

### Detección en Tiempo Real
El modelo YOLO detecta las cartas al instante y las registra en el tablero.
![Detección en Vivo](screenshots/demo_detection.png)

### La Revelación del Oráculo
Una vez reunidas 3 cartas, la IA genera una historia única y la narra con voz.
![Modal de Predicción](screenshots/demo_prediction.png)

---

## 🚀 Características Principales

* **👁️ Visión Artificial (YOLOv12/v8):** Detección de objetos en tiempo real entrenada con un dataset personalizado de cartas de Lotería Mexicana.
* **🧠 IA Generativa (Google Gemini):** Crea narrativas únicas, divertidas y con "jerga mexicana" para interpretar la combinación de cartas.
* **🗣️ Voz Sintética (ElevenLabs):** Convierte el texto generado en una narración de audio dramática y mística al instante.
* **💻 Interfaz Web (Streamlit):** Una experiencia de usuario fluida, responsiva y visualmente atractiva.

---

## 🛠️ Tecnologías Utilizadas

* **Lenguaje:** Python 3.10+
* **Frontend:** Streamlit
* **Computer Vision:** Ultralytics YOLO, OpenCV
* **Generative AI:** Google GenAI SDK (Gemini 1.5 Flash)
* **Text-to-Speech:** ElevenLabs API
* **Despliegue:** Streamlit Cloud

---

## ⚙️ Instalación y Configuración

## NOTA: El proyecto también es accesible desde su página web: https://ai-prediction-project-a7pxtbreuemmdhzksjj9yn.streamlit.app/

Sigue estos pasos para correr el proyecto en tu máquina local:

### 1. Clonar el repositorio
```bash
git clone [https://github.com/tu-usuario/oraculo-loteria.git](https://github.com/tu-usuario/oraculo-loteria.git)
cd oraculo-loteria
```
### 2. Crear un entorno virtual (Recomendado)
```bash
python -m venv venv
# En Windows:
venv\Scripts\activate
# En Mac/Linux:
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Configurar las API Keys
Crea un archivo llamado .env en la raíz del proyecto y agrega tus claves (consíguelas en Google AI Studio y ElevenLabs):
```bash
ELEVENLABS_API_KEY=tu_api_key_aqui
GEMINI_API_KEY=tu_api_key_aqui
```

### 5. Colocar el Modelo
Asegúrate de tener tu archivo de pesos entrenado (best.pt) en la raíz del proyecto.

### 6. Ejecutar la App
```bash
streamlit run app.py
```
