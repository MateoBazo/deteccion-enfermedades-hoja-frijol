"""App web: sube una foto de una hoja de frijol y obtén el diagnóstico.

Ejecutar:  streamlit run app.py
"""
import html
import json
import os
import sys
from pathlib import Path

import streamlit as st
from google.genai import errors
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "src"))
from config import CLASES, DATA, NOMBRES_ES, SALIDAS  # noqa: E402
from predictor import Predictor, superponer  # noqa: E402
from recomendaciones_gemini import recomendar  # noqa: E402

st.set_page_config(page_title="Detector de enfermedades del frijol", layout="wide")

COLOR_CLASE = {"healthy": "#2F6B3F", "angular_leaf_spot": "#A67C00", "bean_rust": "#A4452C"}
COLOR_URGENCIA = {"baja": "#2F6B3F", "media": "#A67C00", "alta": "#A4452C"}

st.markdown("""
<style>
  .block-container {padding-top: 2.2rem; max-width: 1180px;}
  #MainMenu, footer, [data-testid="stDecoration"] {visibility: hidden;}
  h1 {font-weight: 650; letter-spacing: -0.02em; margin-bottom: 0 !important;}
  .sobretitulo {font-size: .78rem; letter-spacing: .12em; text-transform: uppercase; color: #6B756D;}
  .subtitulo {color: #5A645C; margin: .25rem 0 1.6rem 0;}
  .etiqueta {font-size: .75rem; letter-spacing: .08em; text-transform: uppercase; color: #6B756D;}
  .clase {font-size: 1.9rem; font-weight: 650; line-height: 1.2; margin: .15rem 0 .1rem 0;}
  .confianza {color: #5A645C; margin-bottom: 1.2rem;}
  .fila-prob {display: flex; justify-content: space-between; font-size: .9rem; margin-top: .7rem;}
  .barra {height: 6px; background: #E6E9E1; border-radius: 3px; margin-top: .3rem;}
  .barra > div {height: 6px; border-radius: 3px;}
  .insignia {display: inline-block; padding: .15rem .6rem; border-radius: 999px; font-size: .78rem;
             font-weight: 600; color: white; vertical-align: middle; margin-left: .5rem;}
  .tarjeta-titulo {font-weight: 600; margin-bottom: .4rem;}
  .nota {font-size: .8rem; color: #7A847C;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def cargar_predictor():
    return Predictor()


@st.cache_data(show_spinner=False)
def recomendaciones(clase, probs_redondeadas):
    # Cacheadas: misma clase y probabilidades -> no se gasta otra consulta del nivel gratuito
    return recomendar(clase, dict(probs_redondeadas))


def exactitud_test():
    try:
        return json.loads((SALIDAS / "metricas_cnn.json").read_text())["exactitud_test"]
    except (OSError, KeyError, ValueError):
        return None


# --- Encabezado ---
st.markdown('<div class="sobretitulo">Inteligencia Artificial · Visión por computadora</div>',
            unsafe_allow_html=True)
st.title("Detector de enfermedades en hojas de frijol")
st.markdown('<div class="subtitulo">Una red neuronal convolucional analiza la foto; '
            'Gemini genera las recomendaciones de manejo a partir del diagnóstico.</div>',
            unsafe_allow_html=True)

def pie():
    acc = exactitud_test()
    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    st.markdown(f"""<div class="nota" style="border-top:1px solid #DDE1D8; padding-top:.8rem;">
        Modelo: ResNet-18 con transfer learning, entrenada con 1.034 fotos de campo
        {f"· exactitud en test {acc:.1%}" if acc else ""} · clases: sana, mancha angular, roya.</div>""",
                unsafe_allow_html=True)


# --- Entrada de imagen en el centro: arrastrar y soltar ---
ejemplos = sorted((DATA / "test").glob("*/*.jpg"))
etiquetas = {f"{NOMBRES_ES[p.parent.name]} — {p.stem.split('.')[-1]}": p for p in ejemplos}

col_subir, col_ejemplo = st.columns([3, 1.2], gap="medium", vertical_alignment="bottom")
archivo = col_subir.file_uploader("Arrastra aquí la foto de una hoja o haz clic para buscarla",
                                  type=["jpg", "jpeg", "png", "webp"])
ejemplo = col_ejemplo.selectbox("O usa una imagen de prueba", list(etiquetas), index=None,
                                placeholder="Seleccionar…")
usar_llm = col_ejemplo.toggle("Recomendaciones con Gemini", value=bool(os.environ.get("GEMINI_API_KEY")),
                              help="Genera causa, tratamiento y prevención a partir del diagnóstico.")

imagen = Image.open(archivo) if archivo else (Image.open(etiquetas[ejemplo]) if ejemplo else None)

if imagen is None:
    # Zona de arrastre grande mientras no hay imagen
    st.markdown("""<style>
      [data-testid="stFileUploaderDropzone"] {min-height: 220px; border: 2px dashed #B9C2B3;
        background: #F4F6F0; align-items: center; justify-content: center;}
    </style>""", unsafe_allow_html=True)
    st.markdown('<div class="nota">Mejores resultados: una hoja centrada, bien iluminada y enfocada. '
                'Formatos JPG, PNG o WEBP.</div>', unsafe_allow_html=True)
    pie()
    st.stop()

st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)

# --- Diagnóstico de la CNN ---
clase, probs, cam = cargar_predictor().predecir(imagen)

col_img, col_res = st.columns([1.15, 1], gap="large")
with col_img:
    tab_foto, tab_cam = st.tabs(["Foto", "Mapa de atención (Grad-CAM)"])
    tab_foto.image(imagen, width="stretch")
    with tab_cam:
        st.image(superponer(imagen, cam), width="stretch")
        st.markdown('<div class="nota">Las zonas cálidas son las que más influyeron en la predicción.</div>',
                    unsafe_allow_html=True)

with col_res:
    with st.container(border=True):
        barras = "".join(
            f'<div class="fila-prob"><span>{NOMBRES_ES[c]}</span><span>{probs[c]:.1%}</span></div>'
            f'<div class="barra"><div style="width:{probs[c] * 100:.1f}%; background:{COLOR_CLASE[c]}"></div></div>'
            for c in sorted(CLASES, key=probs.get, reverse=True))
        st.markdown(f"""
            <div class="etiqueta">Diagnóstico</div>
            <div class="clase" style="color:{COLOR_CLASE[clase]}">{NOMBRES_ES[clase]}</div>
            <div class="confianza">Confianza del modelo: {probs[clase]:.1%}</div>
            <div class="etiqueta">Probabilidades</div>
            {barras}
        """, unsafe_allow_html=True)
        if probs[clase] < 0.7:
            st.warning("Confianza baja: conviene tomar otra foto más cercana y con mejor luz.")

# --- Recomendaciones de Gemini ---
st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
st.markdown("### Recomendaciones")

if not usar_llm:
    st.markdown('<div class="nota">Activa "Recomendaciones con Gemini" (arriba) para obtener la causa, '
                'el tratamiento y las medidas de prevención según el diagnóstico.</div>', unsafe_allow_html=True)
else:
    with st.spinner("Generando recomendaciones…"):
        try:
            r = recomendaciones(clase, tuple((c, round(p, 2)) for c, p in probs.items()))
        except ValueError:
            st.warning("Falta GEMINI_API_KEY. Copia `.env.example` como `.env` y pon tu clave gratuita.")
            pie()
            st.stop()
        except errors.APIError as e:
            st.error(f"No se pudo consultar a Gemini ({e.code}). Intenta de nuevo en unos segundos.")
            pie()
            st.stop()

    with st.container(border=True):
        st.markdown(f"""
            <div style="font-size:1.05rem; font-weight:600;">{html.escape(r['resumen'])}</div>
            <div style="margin-top:.6rem;"><span class="etiqueta">Urgencia</span>
              <span class="insignia" style="background:{COLOR_URGENCIA[r['urgencia']]}">{r['urgencia'].capitalize()}</span>
            </div>
            <div style="margin-top:.8rem;"><span class="etiqueta">Causa</span><br>{html.escape(r['causa'])}</div>
        """, unsafe_allow_html=True)

    secciones = [("Verificar en campo", "sintomas_a_verificar"), ("Acciones inmediatas", "acciones_inmediatas"),
                 ("Tratamiento", "tratamiento"), ("Prevención", "prevencion")]
    for fila in (secciones[:2], secciones[2:]):
        for col, (titulo, clave) in zip(st.columns(2), fila):
            with col.container(border=True, height="stretch"):
                items = "".join(f"<li>{html.escape(x)}</li>" for x in r[clave])
                st.markdown(f'<div class="tarjeta-titulo">{titulo}</div><ul style="margin:0">{items}</ul>',
                            unsafe_allow_html=True)

    st.markdown(f'<div class="nota">Generado por {r["modelo"]}. Recomendaciones generadas por IA: '
                'confírmelas con un técnico agrícola antes de aplicar productos.</div>', unsafe_allow_html=True)

pie()
