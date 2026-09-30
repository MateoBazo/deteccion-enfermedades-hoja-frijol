# 🌱 Detección de enfermedades en hojas de frijol con IA

**Propósito:** detectar a partir de una foto si una hoja de frijol está **sana** o tiene **mancha angular** o **roya**, dos enfermedades que reducen mucho el rendimiento del cultivo. Detectarlas pronto permite tratar a tiempo y usar menos fungicida.

## Dataset
**Beans** (AI Lab Makerere / NaCRRI, Uganda): 1.295 fotos de campo de 500×500 px, tomadas con celular, en 3 clases balanceadas.

| Clase | Train | Validación | Test |
|---|---|---|---|
| Mancha angular | 345 | 44 | 43 |
| Roya del frijol | 348 | 45 | 43 |
| Sana | 341 | 44 | 42 |

Fuente: https://huggingface.co/datasets/AI-Lab-Makerere/beans

## Enfoque (dos tipos de IA)
1. **CNN con transfer learning:** ResNet-18 preentrenada en ImageNet a la que se le reemplaza la última capa y se reentrena con las fotos de frijol, usando aumento de datos (rotaciones, recortes y cambios de color).
2. **Grad-CAM:** muestra en qué zonas de la imagen se basa la CNN para decidir, para comprobar que mira las lesiones y no el fondo.
3. **LLM (Gemini, nivel gratuito) como asistente agronómico:** la imagen la analiza **solo nuestra CNN**. A Gemini se le envía en texto el diagnóstico y las probabilidades, y genera la causa, qué verificar en campo, las acciones inmediatas, el tratamiento, la prevención y la urgencia. Si la CNN tiene poca confianza, Gemini lo advierte.

## Resultados (CNN, conjunto de test de 128 imágenes nunca vistas)
- **Exactitud: 96,1 %**
- F1 por clase: Sana 0,99 · Mancha angular 0,95 · Roya 0,94
- La mayoría de los errores ocurren entre mancha angular y roya, que se parecen visualmente. Nunca se confundió una hoja enferma con una sana.

Los gráficos están en `outputs/`: distribución de clases, ejemplos, análisis de color, curvas de aprendizaje, matriz de confusión y Grad-CAM.

## Cómo ejecutar
```bash
cd detector-frijol
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# descargar dataset (una vez) en data/train, data/validation, data/test:
for s in train validation test; do
  curl -L -o data/$s.zip https://huggingface.co/datasets/AI-Lab-Makerere/beans/resolve/main/data/$s.zip
  unzip -q data/$s.zip -d data && rm data/$s.zip
done

python src/eda.py                  # 1. análisis exploratorio
python src/entrenar_cnn.py         # 2. entrenar la CNN (~2 min en Mac M1/M2)
python src/gradcam.py              # 3. interpretabilidad
cp .env.example .env               #    poner GEMINI_API_KEY (gratis en aistudio.google.com/apikey)
python src/recomendaciones_gemini.py --imagen data/test/bean_rust/bean_rust_test.0.jpg   # 4. CNN + recomendaciones
streamlit run app.py               # 5. app web
```

## Estructura
```
src/config.py           rutas, clases, transformaciones
src/eda.py              análisis exploratorio
src/entrenar_cnn.py     entrenamiento y evaluación de la CNN
src/predictor.py        inferencia + Grad-CAM
src/gradcam.py          figura de Grad-CAM (aciertos y errores)
src/recomendaciones_gemini.py  recomendaciones de manejo con Gemini
app.py                  interfaz web (Streamlit)
```
