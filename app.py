import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time

# ==========================================
# 1. CONFIGURACIÓN Y "CEREBRO MÍSTICO"
# ==========================================

st.set_page_config(page_title="El Oráculo de la Lotería", page_icon="🔮", layout="wide")

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
    return YOLO("best.pt")

try:
    model = load_model()
except:
    st.error("⚠️ Error: No encuentro el archivo 'best.pt'. Ponlo en la misma carpeta.")
    st.stop()



# ==========================================
# 2. INTERFAZ GRÁFICA (CSS Y ESTILO MEJORADO)
# ==========================================

st.markdown("""
    <style>
    /* Fondo principal con gradiente alegre */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Título principal con animación */
    .main-title {
        font-size: 60px !important;
        font-weight: bold;
        background: linear-gradient(45deg, #FFD700, #FFA500, #FF69B4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        animation: glow 2s ease-in-out infinite;
        text-shadow: 0 0 20px rgba(255, 215, 0, 0.5);
    }
    
    @keyframes glow {
        0%, 100% { filter: brightness(1); }
        50% { filter: brightness(1.3); }
    }
    
    /* Cajas de cartas con efecto hover */
    .card-slot {
        background: linear-gradient(145deg, #ffffff, #f0f0f0);
        border: 3px solid #FFD700;
        padding: 25px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        transition: transform 0.3s, box-shadow 0.3s;
        min-height: 150px;
    }
    
    .card-slot:hover {
        transform: translateY(-10px) rotate(2deg);
        box-shadow: 0 12px 24px rgba(255, 215, 0, 0.4);
    }
    
    /* Resultado final con texto visible */
    .prediction-box {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        padding: 30px;
        border-radius: 20px;
        border: 4px solid #FF69B4;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        animation: slideIn 0.8s ease-out;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(50px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .prediction-title {
        font-size: 40px !important;
        font-weight: bold;
        color: #4B0082 !important;
        text-align: center;
        margin-bottom: 20px;
        text-shadow: 2px 2px 4px rgba(255,255,255,0.5);
    }
    
    .prediction-text {
        font-size: 22px !important;
        color: #1a1a1a !important;
        line-height: 1.8;
        font-weight: 500;
    }
    
    /* Botones personalizados */
    .stButton>button {
        background: linear-gradient(45deg, #FF69B4, #FFD700);
        color: white;
        font-size: 20px;
        font-weight: bold;
        border: none;
        border-radius: 15px;
        padding: 15px 30px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: scale(1.1);
        box-shadow: 0 8px 25px rgba(255, 105, 180, 0.5);
    }
    
    /* Sidebar con estilo */
    .css-1d391kg {
        background: linear-gradient(180deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Emojis animados */
    .emoji-float {
        animation: float 3s ease-in-out infinite;
        display: inline-block;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }
    
    /* Overlay de imagen sobre la cámara */
    [data-testid="stCameraInput"] {
        position: relative;
    }
    
    [data-testid="stCameraInput"]::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: url('oracle_overlay.png');
        background-size: cover;
        background-position: center;
        pointer-events: none;
        z-index: 10;
        opacity: 0.8;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>🔮✨ El Oráculo de la Lotería ✨🔮</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:24px; color:white;'>Muestra <b>3 cartas distintas</b> a la cámara para leer tu destino...</p>", unsafe_allow_html=True)

# Inicializar memoria de cartas encontradas
if 'cartas_vistas' not in st.session_state:
    st.session_state['cartas_vistas'] = []

# ==========================================
# 3. BARRA LATERAL (DATOS TÉCNICOS)
# ==========================================
with st.sidebar:
    st.markdown("### 🧠 Panel Neuronal")
    st.write("🤖 Modelo: YOLOv8 Custom")
    metric_conf = st.empty()
    metric_class = st.empty()
    
    if st.button("🗑️ Reiniciar Lectura"):
        st.session_state['cartas_vistas'] = []
        st.rerun()

# ==========================================
# 4. LÓGICA DE DETECCIÓN
# ==========================================

# Input de cámara
img_file_buffer = st.camera_input("📸 El Ojo que Todo lo Ve")

if img_file_buffer is None:
    st.info("📸 Esperando foto... Haz clic en 'Take Photo' cuando veas 3 cartas diferentes")
else:
    st.success("✅ Foto capturada, analizando...")

if img_file_buffer is not None:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    results = model(cv2_img, conf=0.5)
    
    detectado_ahora = None
    confianza_actual = 0.0

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            confianza = float(box.conf[0])
            
            nombre_detectado = model.names[cls_id] 
            nombre_detectado = nombre_detectado.capitalize() 

            detectado_ahora = nombre_detectado
            confianza_actual = confianza

            metric_conf.metric("🎯 Certeza de Visión", f"{confianza*100:.1f}%")
            metric_class.info(f"👁️ Detectando: {nombre_detectado}")

    if detectado_ahora:
        if detectado_ahora not in st.session_state['cartas_vistas']:
            if detectado_ahora in SIGNIFICADOS:
                st.session_state['cartas_vistas'].append(detectado_ahora)
                st.toast(f"🎉 ¡Carta capturada: {detectado_ahora}!", icon="🃏")
            else:
                st.warning(f"🤔 Veo un {detectado_ahora}, pero no sé qué significa.")

# ==========================================
# 5. MOSTRAR PROGRESO Y RESULTADO
# ==========================================

cartas = st.session_state['cartas_vistas']
total = len(cartas)

st.divider()

# Mostrar slots de las 3 cartas con diseño mejorado
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<h3 style='text-align:center; color:#FFD700;'>🌅 Pasado</h3>", unsafe_allow_html=True)
    if total >= 1:
        st.markdown(f"""
            <div class='card-slot'>
                <p style='font-size:50px; margin:0;'>🎴</p>
                <p style='font-size:28px; font-weight:bold; color:#FF1493; margin:10px 0;'>{cartas[0]}</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("<div class='card-slot'><p style='font-size:24px; color:#999;'>⏳ Esperando...</p></div>", unsafe_allow_html=True)

with col2:
    st.markdown("<h3 style='text-align:center; color:#FFD700;'>⚡ Presente</h3>", unsafe_allow_html=True)
    if total >= 2:
        st.markdown(f"""
            <div class='card-slot'>
                <p style='font-size:50px; margin:0;'>🎴</p>
                <p style='font-size:28px; font-weight:bold; color:#FF1493; margin:10px 0;'>{cartas[1]}</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("<div class='card-slot'><p style='font-size:24px; color:#999;'>⏳ Esperando...</p></div>", unsafe_allow_html=True)

with col3:
    st.markdown("<h3 style='text-align:center; color:#FFD700;'>🌙 Futuro</h3>", unsafe_allow_html=True)
    if total >= 3:
        st.markdown(f"""
            <div class='card-slot'>
                <p style='font-size:50px; margin:0;'>🎴</p>
                <p style='font-size:28px; font-weight:bold; color:#FF1493; margin:10px 0;'>{cartas[2]}</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("<div class='card-slot'><p style='font-size:24px; color:#999;'>⏳ Esperando...</p></div>", unsafe_allow_html=True)

# ==========================================
# 6. LA REVELACIÓN FINAL
# ==========================================

if total >= 3:
    st.divider()
    st.balloons()
    
    c1 = cartas[0]
    c2 = cartas[1]
    c3 = cartas[2]
    
    prediccion = f"""
    <div class='prediction-box'>
        <p class='prediction-title'><span class='emoji-float'>🔮</span> Lectura Final <span class='emoji-float'>🔮</span></p>
        <p class='prediction-text'>
        ✨ En tu pasado, <b style='color:#8B008B;'>{SIGNIFICADOS[c1]}</b> (gracias a <i>{c1}</i>).<br><br>
        🌟 Actualmente, <b style='color:#8B008B;'>{SIGNIFICADOS[c2]}</b>, tal como dicta <i>{c2}</i>.<br><br>
        ⚠️ Pero ten cuidado, porque tu futuro indica que <b style='color:#8B008B;'>{SIGNIFICADOS[c3]}</b>. 
        ¡El <i>{c3}</i> ha hablado!
        </p>
    </div>
    """
    
    st.markdown(prediccion, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    col_center = st.columns([1, 2, 1])
    with col_center[1]:
        if st.button("✨ Leer otra fortuna ✨", use_container_width=True):
            st.session_state['cartas_vistas'] = []
            st.rerun()

elif total > 0:
    st.markdown(f"<p style='text-align:center; font-size:24px; color:white;'>⏳ Sigue mostrando cartas... Faltan <b>{3-total}</b></p>", unsafe_allow_html=True)