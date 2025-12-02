import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time
from textwrap import dedent

# ==========================================
# 1. CONFIGURACIÓN Y "CEREBRO MÍSTICO"
# ==========================================

st.set_page_config(
    page_title="El Oráculo de la Lotería", 
    page_icon="🔮", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Permitir cerrar el modal recargando la página con un query param
params = st.query_params
if params.get("close_modal") == "true":
    st.session_state['cartas_vistas'] = []
    st.session_state['show_modal'] = False
    st.query_params.clear()

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

# Descripciones divertidas para cada carta al detectarla
DESCRIPCIONES = {
    "Apache": "¡Órale! El Apache es un guerrero legendario. Trae energía de batalla, pero cuida de no meterte en pleitos que no son tuyos.",
    "Arana": "¡Uuuy! La Araña... Cuidado con tejer mentiras, porque te puedes enredar solito. O quizás estés tejiendo tu imperio.",
    "Arbol": "¡Perfecto! El Árbol representa estabilidad. Vas a echar raíces donde menos lo esperas. ¡A crecer se ha dicho!",
    "Bandera": "¡Órale! La Bandera es símbolo de patriotismo y valores. Prepárate para defender lo que crees, aunque sea la última dona.",
    "Bandolon": "¡Ay sí! El Bandolón trae música y fiesta. Se viene la pachanga, prepara tus mejores pasos de baile.",
    "Barrilito": "¡Aguas! El Barrilito te advierte que no te pases de copas este fin. O sí, pero no digas que no te avisé.",
    "Botella": "¡Chin! La Botella siempre trae secretos. Una verdad saldrá a flote... o será solo una chela más.",
    "Calavera": "¡No te espantes! La Calavera no es mala, significa transformación. Algo viejo se va, algo nuevo llega. Así es la vida.",
    "Camaron": "¡Ojo vivo! El Camarón dice que el que se duerme, se lo lleva la corriente. ¡Ponte trucha!",
    "Campana": "¡Tan tan! La Campana anuncia noticias importantes. Puede ser buena o mala, pero resonará fuerte.",
    "Catrin": "¡Elegante! El Catrín es todo un galán, pero cuidado, puede ser puro farol. No todo lo que brilla es oro.",
    "Cazo": "¡A cocinar! El Cazo significa que vas a preparar algo importante. Un proyecto, una idea... o unos chilaquiles épicos.",
    "Chalupa": "¡Súbete! La Chalupa trae viajes pequeños pero significativos. Un paseo corto puede cambiarte el día.",
    "Corazon": "¡Ay amor! El Corazón nunca miente. Alguien está pensando en ti... o tú en alguien. Cupido anda cerca.",
    "Corona": "¡Eres el rey/reina! La Corona trae reconocimiento y éxito. Te vas a lucir como nunca.",
    "Cotorro": "¡Shhhh! El Cotorro te recuerda que a veces es mejor quedarse callado. No vayas a echar chisme de más.",
    "Dama": "¡Elegancia pura! La Dama representa a una mujer importante en tu vida. Escucha sus consejos.",
    "Diablito": "¡Ay picarón! El Diablito trae tentaciones. Esa voz en tu cabeza que dice 'dale, no pasa nada'... ¡Cuidado!",
    "Escalera": "¡Pa' arriba! La Escalera significa progreso. Vas a subir, pero paso a paso, sin prisas pero sin pausas.",
    "Estrella": "¡Brillas! La Estrella es la mejor carta. Tienes suerte divina de tu lado. Aprovéchala, campeón.",
    "Gallo": "¡Quiquiriquí! El Gallo te despertará con ideas frescas. Madruga y atrapa esas oportunidades.",
    "Garza": "¡Paciencia! La Garza te enseña que el equilibrio es clave. No te apresures, observa y actúa con calma.",
    "Gorrito": "¡Protégete! El Gorrito significa que debes cuidar tus ideas y pensamientos. No andes compartiendo todo.",
    "Luna": "¡Romántico! La Luna trae secretos nocturnos. Algo misterioso sucederá bajo su luz.",
    "Mano": "¡Te echan la mano! La Mano significa ayuda inesperada. Alguien aparecerá justo cuando lo necesites.",
    "Melon": "¡Dulce vida! El Melón trae sabor y buenos momentos. Disfruta lo bueno que viene.",
    "Muerte": "¡No te asustes! La Muerte es cambio, no final. Algo viejo se va para dar paso a lo nuevo. Es bueno.",
    "Mundo": "¡Todo es tuyo! El Mundo representa éxito total. Tienes el poder de lograr lo que quieras.",
    "Pajaro": "¡Tweet tweet! El Pájaro trae noticias frescas. Alguien te va a buscar o tú buscarás a alguien.",
    "Paraguas": "¡Protección! El Paraguas te cubre de las malas vibras. Eres inmune a la envidia, eres blindado.",
    "Rosa": "¡Qué bonito! La Rosa trae nuevas amistades o amor floreciente. Algo hermoso está creciendo.",
    "Sirena": "¡Aguas! La Sirena canta bonito pero engaña. No te dejes llevar por promesas falsas.",
    "Sol": "¡Qué energía! El Sol te llena de vitalidad. Vas a brillar con luz propia esta semana.",
    "Soldado": "¡Disciplina! El Soldado te recuerda que sin orden no hay progreso. Ponte las pilas.",
    "Tambor": "¡Retumba! El Tambor significa que tus acciones harán ruido. Todo mundo se va a enterar.",
    "Valiente": "¡Échale ganas! El Valiente te dice que enfrentes ese miedo de una vez. Tú puedes.",
    "Venado": "¡Rápido! El Venado es velocidad y astucia. Muévete rápido en los negocios y llegarás lejos.",
    "Violencello": "¡Armonía! El Violoncello trae paz al hogar. La música y la tranquilidad regresan a tu vida."
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
        overflow: hidden;
    }
    
    /* Ocultar elementos de Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Título principal compacto */
    .main-title {
        font-size: 45px !important;
        font-weight: bold;
        background: linear-gradient(45deg, #FFD700, #FFA500, #FF69B4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        animation: glow 2s ease-in-out infinite;
        margin: 10px 0;
    }
    
    @keyframes glow {
        0%, 100% { filter: brightness(1); }
        50% { filter: brightness(1.3); }
    }
    
    /* Cajas de cartas más compactas */
    .card-slot {
        background: linear-gradient(145deg, #ffffff, #f0f0f0);
        border: 3px solid #FFD700;
        padding: 15px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        transition: transform 0.3s, box-shadow 0.3s;
        min-height: 100px;
        height: 100%;
    }
    
    .card-slot:hover {
        transform: translateY(-5px) rotate(2deg);
        box-shadow: 0 12px 24px rgba(255, 215, 0, 0.4);
    }
    
    /* DIÁLOGO MODAL GRANDE CON ANIMACIÓN */
    .modal-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background: rgba(0, 0, 0, 0.85);
        z-index: 9999;
        display: flex;
        align-items: center;
        justify-content: center;
        animation: fadeIn 0.5s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .modal-dialog {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        padding: 35px 40px;
        border-radius: 25px;
        border: 5px solid #FF69B4;
        box-shadow: 0 20px 60px rgba(0,0,0,0.5);
        max-width: 700px;
        width: 85%;
        max-height: 85vh;
        overflow-y: auto;
        animation: scaleIn 0.6s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        position: relative;
    }
    
    @keyframes scaleIn {
        from { 
            opacity: 0; 
            transform: scale(0.3) rotate(-10deg);
        }
        to { 
            opacity: 1; 
            transform: scale(1) rotate(0deg);
        }
    }
    
    .modal-close-x {
        position: absolute;
        top: 15px;
        right: 20px;
        font-size: 32px;
        font-weight: bold;
        color: #4B0082;
        cursor: pointer;
        background: white;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        transition: all 0.3s;
        line-height: 1;
        z-index: 10;
        border: none;
    }
    
    .modal-close-x:hover {
        transform: rotate(90deg) scale(1.1);
        background: #FFD700;
        color: white;
    }
    
    .modal-title {
        font-size: 38px !important;
        font-weight: bold;
        color: #4B0082 !important;
        text-align: center;
        margin-bottom: 25px;
        margin-top: 10px;
        text-shadow: 2px 2px 4px rgba(255,255,255,0.5);
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .modal-text {
        font-size: 20px !important;
        color: #1a1a1a !important;
        line-height: 1.7;
        font-weight: 600;
        text-align: center;
    }
    
    .modal-close-btn {
        margin-top: 20px;
        text-align: center;
    }
    
    /* Botones personalizados */
    .stButton>button {
        background: linear-gradient(45deg, #FF69B4, #FFD700);
        color: white;
        font-size: 18px;
        font-weight: bold;
        border: none;
        border-radius: 15px;
        padding: 12px 25px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: scale(1.1);
        box-shadow: 0 8px 25px rgba(255, 105, 180, 0.5);
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
    
    /* Compactar espaciado */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    
    /* Info compacto */
    .stAlert {
        padding: 8px !important;
        font-size: 14px !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>🔮 El Oráculo de la Lotería 🔮</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px; color:white; margin-bottom:10px;'>Muestra <b>3 cartas distintas</b> para leer tu destino</p>", unsafe_allow_html=True)

# Inicializar memoria de cartas encontradas
if 'cartas_vistas' not in st.session_state:
    st.session_state['cartas_vistas'] = []
if 'show_modal' not in st.session_state:
    st.session_state['show_modal'] = False

# ==========================================
# 3. LAYOUT PRINCIPAL EN DOS COLUMNAS
# ==========================================

col_left, col_right = st.columns([1, 1], gap="medium")

with col_left:
    # Input de cámara
    img_file_buffer = st.camera_input("📸 El Ojo que Todo lo Ve")
    
    # Área de información debajo de la cámara
    info_placeholder = st.empty()
    
    if img_file_buffer is None:
        info_placeholder.info("📸 Captura 3 cartas diferentes", icon="📷")
    
    # LÓGICA DE DETECCIÓN
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

        if detectado_ahora:
            if detectado_ahora not in st.session_state['cartas_vistas']:
                if detectado_ahora in SIGNIFICADOS:
                    st.session_state['cartas_vistas'].append(detectado_ahora)
                    st.toast(f"🎉 ¡Carta capturada: {detectado_ahora}!", icon="🃏")
                    # Mostrar descripción de la carta detectada
                    info_placeholder.success(f"**✨ {detectado_ahora} detectado!**\n\n{DESCRIPCIONES.get(detectado_ahora, 'Una carta misteriosa...')}", icon="🎴")
                else:
                    st.warning(f"🤔 Veo un {detectado_ahora}, pero no sé qué significa.")
            else:
                # Si ya fue detectada antes
                info_placeholder.warning(f"**🔄 {detectado_ahora}** - Ya capturaste esta carta. Muestra una diferente.", icon="⚠️")
        else:
            # No se detectó nada
            info_placeholder.info("🔍 Analizando... Acerca las cartas a la cámara", icon="👀")

with col_right:
    # ==========================================
    # MOSTRAR SLOTS DE CARTAS EN COLUMNA DERECHA
    # ==========================================
    
    cartas = st.session_state['cartas_vistas']
    total = len(cartas)
    
    st.markdown("<h3 style='text-align:center; color:#FFD700; margin-bottom:15px;'>🎴 Cartas Detectadas</h3>", unsafe_allow_html=True)
    
    # Carta 1: Pasado
    st.markdown("<p style='text-align:center; color:#FFD700; font-size:16px; margin:5px 0;'>🌅 Pasado</p>", unsafe_allow_html=True)
    if total >= 1:
        st.markdown(f"""
            <div class='card-slot'>
                <p style='font-size:40px; margin:0;'>🎴</p>
                <p style='font-size:22px; font-weight:bold; color:#FF1493; margin:8px 0;'>{cartas[0]}</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("<div class='card-slot'><p style='font-size:18px; color:#999;'>⏳ Esperando...</p></div>", unsafe_allow_html=True)
    
    # Carta 2: Presente
    st.markdown("<p style='text-align:center; color:#FFD700; font-size:16px; margin:15px 0 5px 0;'>⚡ Presente</p>", unsafe_allow_html=True)
    if total >= 2:
        st.markdown(f"""
            <div class='card-slot'>
                <p style='font-size:40px; margin:0;'>🎴</p>
                <p style='font-size:22px; font-weight:bold; color:#FF1493; margin:8px 0;'>{cartas[1]}</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("<div class='card-slot'><p style='font-size:18px; color:#999;'>⏳ Esperando...</p></div>", unsafe_allow_html=True)
    
    # Carta 3: Futuro
    st.markdown("<p style='text-align:center; color:#FFD700; font-size:16px; margin:15px 0 5px 0;'>🌙 Futuro</p>", unsafe_allow_html=True)
    if total >= 3:
        st.markdown(f"""
            <div class='card-slot'>
                <p style='font-size:40px; margin:0;'>🎴</p>
                <p style='font-size:22px; font-weight:bold; color:#FF1493; margin:8px 0;'>{cartas[2]}</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("<div class='card-slot'><p style='font-size:18px; color:#999;'>⏳ Esperando...</p></div>", unsafe_allow_html=True)
    
    # Botón de reinicio
    if total > 0:
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Reiniciar Lectura", use_container_width=True):
            st.session_state['cartas_vistas'] = []
            st.session_state['show_modal'] = False
            st.rerun()
    
    # Progreso
    if total > 0 and total < 3:
        st.markdown(f"<p style='text-align:center; font-size:16px; color:white; margin-top:15px;'>⏳ Faltan <b>{3-total}</b> carta(s)</p>", unsafe_allow_html=True)

# ==========================================
# MODAL DE REVELACIÓN FINAL
# ==========================================

if total >= 3 and not st.session_state['show_modal']:
    st.session_state['show_modal'] = True
    st.balloons()

if st.session_state['show_modal'] and total >= 3:
    c1 = cartas[0]
    c2 = cartas[1]
    c3 = cartas[2]
    
    # Crear un contenedor para el modal con JavaScript para cerrar
    modal_container = st.empty()
    
    with modal_container.container():
        st.markdown(f"""
        <div class='modal-overlay' id='predictionModal'>
            <div class='modal-dialog'>
                <div class='modal-close-x' id='closeModal'>&times;</div>
                <p class='modal-title'>
                    <span class='emoji-float'>🔮</span> Tu Destino Revelado <span class='emoji-float'>🔮</span>
                </p>
                <div class='modal-text'>
                    <p style='margin:15px 0;'>✨ En tu <b>pasado</b>, {SIGNIFICADOS[c1]}</p>
                    <p style='font-size:18px; color:#4B0082; margin:8px 0;'>(la carta <i>{c1}</i> lo revela)</p>
                    
                    <p style='margin:20px 0 15px 0;'>🌟 En tu <b>presente</b>, {SIGNIFICADOS[c2]}</p>
                    <p style='font-size:18px; color:#4B0082; margin:8px 0;'>(así dicta <i>{c2}</i>)</p>
                    
                    <p style='margin:20px 0 15px 0;'>⚠️ Y en tu <b>futuro</b>, {SIGNIFICADOS[c3]}</p>
                    
                    <p style='font-size:24px; margin-top:25px; font-weight:bold; color:#8B008B;'>¡El <i>{c3}</i> ha hablado!</p>
                </div>
            </div>
        </div>
        
        <script>
        // JavaScript para cerrar el modal
        const closeBtn = document.getElementById('closeModal');
        const modal = document.getElementById('predictionModal');
        
        if (closeBtn) {{
            closeBtn.onclick = function() {{
                window.parent.postMessage({{
                    type: 'streamlit:setComponentValue',
                    key: 'close_modal',
                    value: true
                }}, '*');
            }}
        }}
        
        if (modal) {{
            modal.onclick = function(event) {{
                if (event.target === modal) {{
                    window.parent.postMessage({{
                        type: 'streamlit:setComponentValue',
                        key: 'close_modal',
                        value: true
                    }}, '*');
                }}
            }}
        }}
        </script>
        """, unsafe_allow_html=True)
        
        # Botón para cerrar el modal
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("✨ Leer otra fortuna ✨", key="close_modal_btn", use_container_width=True):
                st.session_state['cartas_vistas'] = []
                st.session_state['show_modal'] = False
                st.rerun()