import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from textwrap import dedent
import logging
from google import genai
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
import os
from dotenv import load_dotenv
import random

load_dotenv()

def get_secret(key_name):
    """Busca la clave primero en Streamlit Cloud Secrets, luego en variables de entorno"""
    if key_name in st.secrets:
        return st.secrets[key_name]
    else:
        return os.getenv(key_name)


ELEVENLABS_API_KEY = get_secret("ELEVENLABS_API_KEY")
#ELEVENLABS_API_KEY = "fake api"
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")

# Validar que las keys existen
if not ELEVENLABS_API_KEY or not GEMINI_API_KEY:
    st.error("❌ Error: No se encontraron las API keys. Crea un archivo .env con tus credenciales.")
    st.info("""
    Crea un archivo `.env` en la carpeta del proyecto con este contenido:
    ```
    ELEVENLABS_API_KEY=tu_key_aqui
    GEMINI_API_KEY=tu_key_aqui
    ```
    """)
    st.stop()


# Configurar clientes
client_gemini = genai.Client(api_key=GEMINI_API_KEY)
client_eleven = ElevenLabs(api_key=ELEVENLABS_API_KEY)

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@st.cache_data(show_spinner=False, ttl=3600)
def generar_prediccion_ia(c1, c2, c3):
    """
    Genera una historia coherente y fluida conectando las 3 cartas.
    """
    logger.info(f"🤖 Generando narrativa para: {c1} -> {c2} -> {c3}")
    
    # PROMPT DE INGENIERÍA NARRATIVA
    # El truco aquí es pedirle que actúe como un personaje y prohibirle estructuras rígidas.
    prompt = f"""
    Actúa como un brujo místico de feria mexicana, sabio pero con jerga de barrio.
    
    Tienes 3 cartas de la lotería que representan la línea temporal de una persona:
    1. PASADO (Causa): {c1}
    2. PRESENTE (Situación actual): {c2}
    3. FUTURO (Consecuencia/Advertencia): {c3}
    
    TU TAREA:
    Escribe UNA SOLA predicción de máximo 100 palabras que conecte estas tres cartas en una historia fluida.
    
    REGLAS DE ORO:
    - NO empieces las oraciones con "Tu pasado fué", "Tu presente es" o "Tu futuro será". Usa conectores como "antes", "ahorita", "por eso", "así que aguas".
    - NO hagas listas. Debe ser un párrafo corrido.
    - Menciona las cartas por su nombre.
    - Tono: Divertido, místico.
    - Termina con una advertencia o consejo contundente basado en la tercera carta.
    - Crea historias coherentes.
    Ejemplo de estilo deseado:
    "Uy, se ve que el Apache te trajo problemas, y aunque ahorita el Gallo te tiene muy despierto y movido, bájale dos rayitas porque la Sirena te quiere endulzar el oído con mentiras."
    """
    
    try:
        response = client_gemini.models.generate_content(
            model='gemini-2.5-flash', 
            contents=prompt,
            config={'temperature': 1.0} # Alta temperatura para más creatividad
        )
        
        texto = response.text.strip()
        # Limpieza extra
        texto = texto.replace('"', '').replace('*', '')
        return texto

    except Exception as e:
        logger.error(f"❌ Error Gemini: {e}")
        # Fallback genérico pero fluido
        return f"Vaya combinación. El {c1} dejó huella, ahora el {c2} marca tu paso, ¡pero cuidado con el {c3} que viene fuerte!"

@st.cache_data(show_spinner=False)
def texto_a_audio_elevenlabs(texto_prediccion):
    """
    Genera audio natural uniendo una intro aleatoria + la predicción fluida.
    """
    # 1. Seleccionamos una intro al azar para variedad
    intro = random.choice(INTROS_DRAMATICAS)
    
    # 2. Unimos el texto completo
    texto_final = f"{intro} ... {texto_prediccion}"
    
    logger.info(f"🎤 Generando voz para: '{texto_final[:40]}...'")
    
    try:
        # Usamos settings probados para que suene expresivo pero estable
        response = client_eleven.text_to_speech.convert(
            voice_id="TX3LPaxmHKxFdv7VOQHJ", # Arnold (Voz profunda/mística)
            #optimize_streaming_latency="0",
            output_format="mp3_44100_128",
            text=texto_final,
            model_id="eleven_v3",
            voice_settings=VoiceSettings(
                stability=0.4,       # Un poco más bajo = más emoción/variación
                similarity_boost=0.8, # Mantiene la identidad de la voz
                style=0.6,           # Estilo dramático moderado
                use_speaker_boost=True
            )
        )
        audio_bytes = b"".join(response)
        return audio_bytes, texto_final # Devolvemos también el texto para mostrarlo si quieres
    except Exception as e:
        logger.error(f"❌ Error ElevenLabs: {e}")
        return None, None


# Intros aleatorias para que no suene repetitivo
INTROS_DRAMATICAS = [
    "¡Pongan mucha atención!",
    "¡Híjole! Las cartas están calientes.",
    "¡Escucha bien lo que dice el destino!",
    "¡Ay nanita! Mira nomás lo que salió.",
    "¡Órale! El oráculo ha hablado.",
    "Silencio todos, que las cartas revelan la verdad."
]

# Inicializar memoria de cartas encontradas
if 'cartas_vistas' not in st.session_state:
    st.session_state['cartas_vistas'] = []
if 'show_modal' not in st.session_state:
    st.session_state['show_modal'] = False
if 'camera_reset_counter' not in st.session_state:
    st.session_state['camera_reset_counter'] = 0

# ==========================================
# 1. CONFIGURACIÓN Y "CEREBRO MÍSTICO"
# ==========================================
st.set_page_config(
    page_title="El Oráculo de la Lotería", 
    page_icon="🔮", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

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



# Cargar Modelo
@st.cache_resource
def load_model():
    logger.info("Cargando modelo YOLO...")
    return YOLO("best.pt")

try:
    model = load_model()
    logger.info("✅ Modelo cargado exitosamente.")
except Exception as e:
    logger.critical(f"❌ Error fatal cargando el modelo: {e}")
    st.error("⚠️ Error: No encuentro el archivo 'best.pt'. Ponlo en la misma carpeta.")
    st.stop()

# ==========================================
# 2. INTERFAZ GRÁFICA (CSS Y ESTILO)
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
    # Input de cámara con key dinámica para forzar reset
    camera_key = f"camera_{st.session_state.get('camera_reset_counter', 0)}"
    img_file_buffer = st.camera_input("📸 El Ojo que Todo lo Ve", key=camera_key)
    
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
                    logger.info(f"Carta detectada: {detectado_ahora}")
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
            # Limpiar cartas
            st.session_state['cartas_vistas'] = []
            st.session_state['show_modal'] = False
            # Incrementar contador para resetear la cámara
            st.session_state['camera_reset_counter'] = st.session_state.get('camera_reset_counter', 0) + 1
            st.rerun()
    
    # Progreso
    if total > 0 and total < 3:
        st.markdown(f"<p style='text-align:center; font-size:16px; color:white; margin-top:15px;'>⏳ Faltan <b>{3-total}</b> carta(s)</p>", unsafe_allow_html=True)

# ==========================================
# 4. MODAL DE REVELACIÓN FINAL (CON VOZ Y LOGS 🎙️)
# ==========================================
@st.dialog("🔮 Tu Destino Revelado 🔮")
def mostrar_revelacion(c1, c2, c3):
    st.markdown("""
    <style>
    .pred-title { font-size: 22px; font-weight: bold; color: #FFD700; margin: 20px 0 15px 0; text-align: center; }
    .pred-text { font-size: 20px; color: #f0f0f0; margin-bottom: 20px; line-height: 1.6; font-weight: 400; text-align: center; }
    .final-destiny { font-size: 26px; font-weight: bold; color: #C71585; text-align: center; margin-top: 30px; padding: 20px; background-color: #FFF0F5; border-radius: 12px; border: 2px dashed #C71585; box-shadow: 0 0 15px rgba(199, 21, 133, 0.4); }
    </style>
    """, unsafe_allow_html=True)

    # 1. GENERAR PREDICCIÓN CON IA
    st.markdown("<div class='pred-title'>🔮 El Oráculo Consulta las Cartas...</div>", unsafe_allow_html=True)
    
    with st.spinner("✨ Interpretando el destino..."):
        prediccion_ia = generar_prediccion_ia(c1, c2, c3)
        resultado_audio = texto_a_audio_elevenlabs(prediccion_ia)
    
    # 2. MOSTRAR PREDICCIÓN
    st.markdown(f"<div class='pred-text'>{prediccion_ia}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='final-destiny'>¡Las cartas {c1}, {c2} y {c3} han hablado!</div>", unsafe_allow_html=True)

    # 3. Reproducir AUDIO
    if resultado_audio and resultado_audio[0]:  # resultado_audio es (audio_bytes, texto_final)
        audio_bytes, texto_completo = resultado_audio
        st.audio(audio_bytes, format='audio/mp3', autoplay=True)
        logger.info(f"✅ Audio reproducido: '{texto_completo[:50]}...'")
    else:
        st.warning("🔇 El oráculo está afónico, pero tu destino está escrito arriba.")
        logger.warning("Fallo en audio")

    # 4. BOTÓN REINICIO
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("✨ Leer otra fortuna ✨", type="primary", use_container_width=True):
        # Limpiar cartas
        st.session_state['cartas_vistas'] = [] 
        st.session_state['show_modal'] = False
        # Incrementar contador para resetear la cámara
        st.session_state['camera_reset_counter'] = st.session_state.get('camera_reset_counter', 0) + 1
        st.rerun()

# Lógica de disparo del modal
if total >= 3:
    if not st.session_state['show_modal']:
        st.session_state['show_modal'] = True
        st.balloons()
        st.rerun()
    
    if st.session_state['show_modal']:
        mostrar_revelacion(cartas[0], cartas[1], cartas[2])