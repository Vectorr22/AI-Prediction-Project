import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time

# ==========================================
# 1. CONFIGURACIÓN Y "CEREBRO MÍSTICO"
# ==========================================

st.set_page_config(page_title="El Oráculo de la Lotería", page_icon="🔮")

# Diccionario con significados divertidos/místicos para cada carta

SIGNIFICADOS = {
    "Apache": "enfrentarás un conflicto ajeno",
    "Arana": "tejerás una red de mentiras (o de éxito)",
    "Arbol": "echarás raíces donde menos lo esperas",
    "Bandera": "tendrás que defender tus ideales",
    "Bandolon": "vendrá música y fiesta a tu vida",
    "Barrilito": "cuidado con los excesos este fin de semana",
    "Botella": "una verdad saldrá a la luz (o una bebida)",
    "Calavera": "un cambio radical y necesario se acerca",
    "Camaron": "si te duermes, te llevará la corriente",
    "Campana": "recibirás una noticia resonante",
    "Catrin": "conocerás a alguien elegante pero engañoso",
    "Cazo": "cocinarás un proyecto importante",
    "Chalupa": "un viaje pequeño te cambiará el ánimo",
    "Corazon": "el amor tocará a tu puerta (o la de tu vecino)",
    "Corona": "recibirás el reconocimiento que mereces",
    "Cotorro": "cuidado con hablar de más",
    "Dama": "una mujer influyente te ayudará",
    "Diablito": "una tentación pondrá a prueba tu voluntad",
    "Escalera": "subirás de nivel, pero paso a paso",
    "Estrella": "tienes una guía divina, confía en tu suerte",
    "Gallo": "te despertarás temprano con nuevas ideas",
    "Garza": "necesitas equilibrio y paciencia",
    "Gorrito": "tendrás que proteger tus ideas",
    "Luna": "secretos románticos bajo la noche",
    "Mano": "recibirás ayuda inesperada",
    "Melon": "la vida será dulce contigo",
    "Muerte": "deja ir lo viejo para que entre lo nuevo",
    "Mundo": "el éxito global está en tus manos",
    "Pajaro": "noticias vuelan hacia ti",
    "Paraguas": "protégete de las malas vibras",
    "Rosa": "florecerá una nueva amistad",
    "Sirena": "no te dejes llevar por cantos falsos",
    "Sol": "energía y vitalidad llenarán tu semana",
    "Soldado": "necesitas disciplina para lograr tu meta",
    "Tambor": "tus pasos harán mucho ruido",
    "Valiente": "enfrenta ese miedo ahora mismo",
    "Venado": "se rápido y astuto en los negocios",
    "Violencello": "la armonía regresará a tu hogar"
}

# Cargar Modelo (con caché para que no recargue lento)
@st.cache_resource
def load_model():
    return YOLO("best.pt") # <--- ASEGURATE QUE TU MODELO SE LLAME ASÍ

try:
    model = load_model()
except:
    st.error("⚠️ Error: No encuentro el archivo 'best.pt'. Ponlo en la misma carpeta.")
    st.stop()

# ==========================================
# 2. INTERFAZ GRÁFICA (CSS Y ESTILO)
# ==========================================

st.markdown("""
    <style>
    .big-font { font-size:30px !important; font-weight: bold; color: #FF4B4B; }
    .card-box { border: 2px solid #FF4B4B; padding: 10px; border-radius: 10px; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

st.title("🔮 El Oráculo de la Lotería")
st.markdown("Muestra **3 cartas distintas** a la cámara para leer tu destino...")

# Inicializar memoria de cartas encontradas
if 'cartas_vistas' not in st.session_state:
    st.session_state['cartas_vistas'] = []

# ==========================================
# 3. BARRA LATERAL (DATOS TÉCNICOS)
# ==========================================
with st.sidebar:
    st.header("🧠 Panel Neuronal")
    st.write("Modelo: YOLOv8 Custom")
    metric_conf = st.empty() # Placeholder para actualizar
    metric_class = st.empty()
    
    if st.button("🗑️ Reiniciar Lectura"):
        st.session_state['cartas_vistas'] = []
        st.rerun()

# ==========================================
# 4. LÓGICA DE DETECCIÓN
# ==========================================

# Input de cámara
img_file_buffer = st.camera_input("El Ojo que Todo lo Ve")

if img_file_buffer is None:
    st.info("📸 Esperando foto... Haz clic en 'Take Photo' cuando veas 3 cartas diferentes")
else:
    st.success("✅ Foto capturada, analizando...")

if img_file_buffer is not None:
    # Convertir imagen para OpenCV
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # Predicción
    results = model(cv2_img, conf=0.5) # Confianza mínima 50%
    
    detectado_ahora = None
    confianza_actual = 0.0

    # Analizar resultados
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            confianza = float(box.conf[0])
            
            # Obtener nombre (Usamos el diccionario interno del modelo o nuestra lista)
            # YOLO suele devolver nombres en minúscula, ajustamos
            nombre_detectado = model.names[cls_id] 
            
            # Normalizar nombre (Capitalizar primera letra: garza -> Garza)
            nombre_detectado = nombre_detectado.capitalize() 

            detectado_ahora = nombre_detectado
            confianza_actual = confianza

            # Actualizar barra lateral (Efecto Matrix)
            metric_conf.metric("Certeza de Visión", f"{confianza*100:.1f}%")
            metric_class.info(f"Detectando: {nombre_detectado}")

    # Lógica de Acumulación (Solo guardar si no la hemos visto antes)
    if detectado_ahora:
        if detectado_ahora not in st.session_state['cartas_vistas']:
            # Solo guardamos si está en nuestra lista de significados (filtro de seguridad)
            if detectado_ahora in SIGNIFICADOS:
                st.session_state['cartas_vistas'].append(detectado_ahora)
                st.toast(f"¡Carta capturada: {detectado_ahora}!", icon="🃏")
            else:
                # Si el modelo detecta algo que no tenemos definido (raro, pero posible)
                st.warning(f"Veo un {detectado_ahora}, pero no sé qué significa.")

# ==========================================
# 5. MOSTRAR PROGRESO Y RESULTADO
# ==========================================

cartas = st.session_state['cartas_vistas']
total = len(cartas)

st.divider()

# Mostrar slots de las 3 cartas
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Pasado")
    if total >= 1:
        st.success(f"🎴 {cartas[0]}")
    else:
        st.info("Esperando...")

with col2:
    st.markdown("### Presente")
    if total >= 2:
        st.success(f"🎴 {cartas[1]}")
    else:
        st.info("Esperando...")

with col3:
    st.markdown("### Futuro")
    if total >= 3:
        st.success(f"🎴 {cartas[2]}")
    else:
        st.info("Esperando...")

# ==========================================
# 6. LA REVELACIÓN FINAL
# ==========================================

if total >= 3:
    st.divider()
    st.balloons() # <--- EFECTO WOW
    
    c1 = cartas[0]
    c2 = cartas[1]
    c3 = cartas[2]
    
    # Construir la frase
    prediccion = f"""
    <div style='background-color:#f0f2f6; padding:20px; border-radius:10px;'>
        <p class='big-font'>🔮 Lectura Final:</p>
        <p style='font-size:20px;'>
        En tu pasado, <b>{SIGNIFICADOS[c1]}</b> (gracias a <i>{c1}</i>).<br><br>
        Actualmente, <b>{SIGNIFICADOS[c2]}</b>, tal como dicta <i>{c2}</i>.<br><br>
        Pero ten cuidado, porque tu futuro indica que <b>{SIGNIFICADOS[c3]}</b>. 
        ¡El <i>{c3}</i> ha hablado!
        </p>
    </div>
    """
    
    st.markdown(prediccion, unsafe_allow_html=True)
    
    if st.button("✨ Leer otra fortuna"):
        st.session_state['cartas_vistas'] = []
        st.rerun()

elif total > 0:
    st.write(f"Sigue mostrando cartas... Faltan {3-total}")