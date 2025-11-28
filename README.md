# Trabajo Práctico 2 - Procesamiento de Imágenes

Este repositorio contiene las soluciones para los dos ejercicios del Trabajo Práctico N° 2 de Procesamiento de Imágenes.  
Incluye los scripts `parte1.py`, `parte2.py` y el informe técnico `INFORME PDI TP2.pdf`.

## 📋 Prerrequisitos

* **Python 3**: Asegúrate de tener Python 3 instalado en tu sistema.  
  Puedes descargarlo desde https://www.python.org/.

---

## ⚙️ Configuración del Entorno

Se recomienda utilizar un entorno virtual para gestionar las dependencias del proyecto.

1.  **Crear el entorno virtual:**  
    Abre una terminal o línea de comandos en la carpeta del proyecto y ejecuta:
    ```bash
    python -m venv .venv
    ```
    *(Reemplaza `.venv` con el nombre que prefieras para tu entorno si lo deseas).*

2.  **Activar el entorno virtual:**
    * **En Windows:**
        ```bash
        .\.venv\Scripts\activate
        ```
    * **En macOS/Linux:**
        ```bash
        source .venv/bin/activate
        ```
    Verás el nombre del entorno (ej. `(.venv)`) al principio de la línea de comandos, indicando que está activo.

3.  **Instalar las dependencias:**  
    Con el entorno activado, instala las bibliotecas necesarias:
    ```bash
    pip install numpy matplotlib opencv-contrib-python
    ```

---

## ▶️ Ejecución de los Scripts

### Ejercicio 1 — Detección y Clasificación de Monedas + Conteo de Dados  
**Archivo:** `parte1.py`

Este script procesa la imagen `monedas.jpg`, que contiene **monedas y dados** sobre un fondo no uniforme.  
El algoritmo realiza tres tareas principales:

#### A. Segmentación
- Conversión a escala de grises.  
- Filtro de mediana (kernel 9) para reducir ruido sin perder bordes.  
- Umbralización y búsqueda de contornos.  
- Detección de círculos con Transformada de Hough (`cv2.HoughCircles`).

#### B. Clasificación y Conteo de Monedas
La clasificación se realiza según el radio detectado por Hough:

| Tipo de moneda | Condición de radio |
|----------------|--------------------|
| Pequeñas (10 ¢) | r < 145 |
| Medianas (1 $)  | 145 ≤ r < 170 |
| Grandes (50 ¢)  | r ≥ 170 |

Se muestran las monedas detectadas con colores (Azul: pequeñas, Verde: medianas, Rojo: grandes).

#### C. Detección y Conteo de Dados
- Umbralización binaria para aislar cuerpos de dados (thresh = 170).  
- Búsqueda de contornos y filtrado por área (> 35000), aproximación poligonal (4 vértices) y relación de aspecto (0.7–1.3).  
- Para cada dado detectado se aísla la ROI y se cuentan los puntos oscuros mediante HoughCircles (parámetros ajustados para radios pequeños).  
- Resultado: conteo de dados y valor (número de puntos) por cara.

#### ▶️ Ejecutar:
```bash
python parte1.py
```

El script mostrará varias figuras con:

* Imagen original y preprocesada  
* Monedas detectadas y clasificadas  
* Dados detectados y conteo de puntos (con anotaciones)

---

## Ejercicio 2 — Detección de Patentes y Segmentación de Caracteres  
**Archivo:** `parte2.py`

Este script procesa 12 imágenes (`img01.png` a `img12.png`) con vehículos y patentes.  
El objetivo es localizar la placa patente y segmentar sus caracteres.

### 1. Preprocesamiento y detección de candidatos

* Lectura en escala de grises.  
* Umbralización de Otsu (`cv2.THRESH_OTSU`) y bitwise-not para resaltar caracteres.  
* `cv2.findContours` para obtener candidatos (contornos).

### 2. Filtrado inicial de candidatos

Se filtran contornos por:

* Relación de aspecto (h/w) entre **1.5 y 3.0**.  
* Área entre **30 y 500** píxeles.

### 3. Agrupación lógica: `filtrar_por_agrupacion`

Función heurística que agrupa candidatos ordenados por X, comprobando:

* Similitud de altura entre caracteres.  
* Alineación vertical de centros.  
* Proximidad horizontal razonable.

Se selecciona el **mejor grupo** (más elementos y tamaño relevante) como la placa.

### 4. Extracción y segmentación de caracteres

* Se calcula el bounding box del grupo ganador y se aplica un **padding de 15 px** para recortar la patente.  
* Cada carácter se recorta sobre la imagen en escala de grises y se ordena de izquierda a derecha.  
* Se genera una tira horizontal con todos los caracteres segmentados para visualización.

### ▶️ Ejecutar:
```bash
python parte2.py
```

El script genera figuras con:

* Paso 1: Binarización (Otsu) para cada imagen  
* Paso 2: Candidatos detectados (rectángulos)  
* Paso 3: Recorte final de la placa y estado (Detectado / No Detectado)  
* Visualización final con caracteres segmentados por imagen

---

## 📄 Informe en PDF

El archivo **INFORME PDI TP2.pdf** incluye:

* Descripción completa de ambos ejercicios  
* Problemas enfrentados  
* Técnicas implementadas (Hough, Otsu, contornos, heurísticas)  
* Capturas de pantalla de los pasos intermedios  
* Conclusiones finales

---

## 📴 Desactivar el Entorno

Cuando termines de trabajar, puedes desactivar el entorno virtual simplemente ejecutando:

```bash
deactivate
```
