"""Paso 4: Recomendaciones con Gemini a partir del diagnóstico de la CNN.

La imagen la analiza SOLO nuestra CNN. A Gemini se le envía en texto el resultado
(clase y probabilidades) y genera la explicación y las soluciones para el agricultor.

Uso:
  python src/recomendaciones_gemini.py --imagen data/test/bean_rust/bean_rust_test.0.jpg
"""
import argparse
import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from config import CLASES, NOMBRES_ES, RAIZ

load_dotenv(RAIZ / ".env")
# Se prueban en orden: si uno está saturado (503), sin cupo (429) o retirado (404), pasa al siguiente
MODELOS = [os.environ.get("GEMINI_MODEL", "gemini-3.8-flash"), "gemini-flash-latest",
           "gemini-3.5-flash", "gemini-flash-lite-latest"]

SISTEMA = """Eres un ingeniero agrónomo especialista en el cultivo de frijol (Phaseolus vulgaris).
Recibes el resultado de un modelo de visión artificial que analizó la foto de una hoja y
debes aconsejar a un pequeño agricultor. Condiciones posibles:
- Mancha angular (Pseudocercospora griseola)
- Roya del frijol (Uromyces appendiculatus)
- Sana
Reglas:
- Responde en español claro y práctico, sin tecnicismos innecesarios.
- Prioriza el manejo integrado: prácticas culturales primero y químicos solo si hace falta,
  indicando el tipo de producto (ingrediente activo), no marcas comerciales.
- Recuerda respetar las dosis de la etiqueta y usar equipo de protección.
- Si la confianza del modelo es baja, dilo y explica cómo confirmar el diagnóstico."""

ESQUEMA = {
    "type": "object",
    "properties": {
        "resumen": {"type": "string", "description": "1-2 frases con el diagnóstico en lenguaje simple"},
        "causa": {"type": "string", "description": "agente causal y condiciones que favorecen la enfermedad"},
        "sintomas_a_verificar": {"type": "array", "items": {"type": "string"},
                                 "description": "qué debe revisar el agricultor en campo para confirmar"},
        "acciones_inmediatas": {"type": "array", "items": {"type": "string"}},
        "tratamiento": {"type": "array", "items": {"type": "string"}},
        "prevencion": {"type": "array", "items": {"type": "string"}},
        "urgencia": {"type": "string", "enum": ["baja", "media", "alta"]},
    },
    "required": ["resumen", "causa", "sintomas_a_verificar", "acciones_inmediatas",
                 "tratamiento", "prevencion", "urgencia"],
}


def nivel_confianza(p: float) -> str:
    return "alta" if p >= 0.9 else "media" if p >= 0.7 else "baja"


def recomendar(clase: str, probs: dict[str, float], cliente: genai.Client | None = None) -> dict:
    """Pide a Gemini recomendaciones para el diagnóstico dado por la CNN."""
    cliente = cliente or genai.Client()  # lee GEMINI_API_KEY del entorno
    detalle = ", ".join(f"{NOMBRES_ES[c]} {probs[c]:.0%}" for c in CLASES)
    prompt = (f"Diagnóstico del modelo de visión: {NOMBRES_ES[clase]}.\n"
              f"Probabilidades: {detalle}.\n"
              f"Confianza: {nivel_confianza(probs[clase])}.\n"
              "Genera las recomendaciones para el agricultor.")
    config = types.GenerateContentConfig(
        system_instruction=SISTEMA,
        response_mime_type="application/json",
        response_json_schema=ESQUEMA,
        temperature=0.3,
    )
    for modelo in dict.fromkeys(MODELOS):  # sin repetidos, respetando el orden
        try:
            respuesta = cliente.models.generate_content(model=modelo, contents=prompt, config=config)
        except errors.APIError as e:
            if e.code in (404, 429, 503) and modelo != MODELOS[-1]:
                print(f"[Gemini] {modelo} no disponible ({e.code}), probando el siguiente…")
                continue
            raise
        return {**json.loads(respuesta.text), "modelo": modelo}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagen", required=True, help="foto de la hoja a diagnosticar")
    args = ap.parse_args()
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("Falta GEMINI_API_KEY. Copia .env.example como .env y pon tu clave.")

    from PIL import Image
    from predictor import Predictor

    clase, probs, _ = Predictor().predecir(Image.open(args.imagen))
    print(f"CNN: {NOMBRES_ES[clase]} ({probs[clase]:.1%})\n")
    print(json.dumps(recomendar(clase, probs), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
