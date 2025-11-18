import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

# --- LÓGICA DE AGRUPACIÓN ---
def filtrar_por_agrupacion(candidatos_rects) -> list: 
    if not candidatos_rects:
        return []
    candidatos_ordenados = sorted(candidatos_rects, key=lambda r: r[0])
    grupos = []
    
    for rect in candidatos_ordenados:
        x, y, w, h = rect
        agregado = False
        for grupo in grupos:
            last_x, last_y, last_w, last_h = grupo[-1]
            diff_h = abs(h - last_h) / float(max(h, last_h))
            centro_y_actual = y + h/2
            centro_y_last = last_y + last_h/2
            diff_y = abs(centro_y_actual - centro_y_last) / float(max(h, last_h))
            distancia_x = x - (last_x + last_w)
            
            if diff_h < 0.6 and diff_y < 0.4 and -5 < distancia_x < (w * 2.3):
                grupo.append(rect)
                agregado = True
                break
        if not agregado:
            grupos.append([rect])

    grupos_validos = [g for g in grupos if len(g) >= 3]
    if not grupos_validos:
        return []
    mejor_grupo = sorted(grupos_validos,
                         key=lambda g: (len(g), g[0][2] * g[0][3]),
                         reverse=True)[0]
    return mejor_grupo


# --- PROCESAMIENTO COMPLETO ---
def procesar_patente_completo(ruta_imagen):
    img_gray = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        return None, None, None, None, None, "Error de carga"

    # Paso 1
    _, img_binary = cv2.threshold(img_gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_invertida = cv2.bitwise_not(img_binary)

    # Paso 2
    contornos, _ = cv2.findContours(img_invertida, cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE)
    candidatos_crudos = []
    img_candidatos = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(h) / w
        area = w * h
        if 0.8 <= aspect_ratio <= 5.0 and 30 < area < 5000:
            candidatos_crudos.append((x, y, w, h))
            cv2.rectangle(img_candidatos, (x, y), (x+w, y+h),
                          (255, 0, 0), 2)

    # Paso 3
    grupo = filtrar_por_agrupacion(candidatos_crudos)
    
    if grupo:
        min_x = min([r[0] for r in grupo])
        min_y = min([r[1] for r in grupo])
        max_x = max([r[0] + r[2] for r in grupo])
        max_y = max([r[1] + r[3] for r in grupo])

        pad = 15
        h_img, w_img = img_gray.shape
        crop_x = max(0, min_x - pad)
        crop_y = max(0, min_y - pad)
        crop_w = min(w_img, max_x + pad)
        crop_h = min(h_img, max_y + pad)

        recorte = img_gray[crop_y:crop_h, crop_x:crop_w]  # GRIS REAL
        recorte_color = cv2.cvtColor(recorte, cv2.COLOR_GRAY2BGR)

        grupo_ajustado = [(x - crop_x, y - crop_y, w, h)
                          for (x, y, w, h) in grupo]

        for (x, y, w, h) in grupo_ajustado:
            cv2.rectangle(recorte_color, (x, y), (x+w, y+h),
                          (0, 255, 0), 1)

        estado = f"Detectado ({len(grupo)})"
    else:
        recorte_color = np.zeros((50, 150, 3), dtype=np.uint8)
        recorte = np.zeros((50, 150), dtype=np.uint8)
        grupo_ajustado = []
        estado = "No Detectado"

    return img_invertida, img_candidatos, recorte_color, grupo_ajustado, recorte, estado



# --- CARGA DE IMÁGENES ---
datos_graficos = []
for i in range(1, 13):
    nombre = f"img{i:02d}.png"
    img_bin, img_cand, img_final, grupo_adj, recorte_gris, estado = procesar_patente_completo(nombre)

    if img_bin is not None:
        datos_graficos.append((nombre, img_bin, img_cand, img_final,
                               grupo_adj, recorte_gris, estado))



# --- VISUALIZACIÓN PASO 1 — 3 ---
rows, cols = 3, 4


# Paso 1
fig1, axes1 = plt.subplots(rows, cols, figsize=(16, 8))
fig1.suptitle("Paso 1: Binarización y Thresholding")
axes1 = axes1.flatten()

for idx, ax in enumerate(axes1):
    if idx < len(datos_graficos):
        nombre, img_bin, _, _, _, _, _ = datos_graficos[idx]
        ax.imshow(img_bin, cmap='gray')
        ax.set_title(nombre)
    ax.axis('off')


# Paso 2
fig2, axes2 = plt.subplots(rows, cols, figsize=(16, 8))
fig2.suptitle("Paso 2: Contornos candidatos")
axes2 = axes2.flatten()

for idx, ax in enumerate(axes2):
    if idx < len(datos_graficos):
        nombre, _, img_cand, _, _, _, _ = datos_graficos[idx]
        ax.imshow(cv2.cvtColor(img_cand, cv2.COLOR_BGR2RGB))
        ax.set_title(nombre)
    ax.axis('off')


# Paso 3
fig3, axes3 = plt.subplots(rows, cols, figsize=(16, 6))
fig3.suptitle("Paso 3: Recorte final detectado")
axes3 = axes3.flatten()

for idx, ax in enumerate(axes3):
    if idx < len(datos_graficos):
        nombre, _, _, img_final, _, _, estado = datos_graficos[idx]
        ax.imshow(cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{nombre}\n{estado}")
    ax.axis('off')


plt.tight_layout()
plt.show()



# --- FUNCIÓN DE SEGMENTACIÓN DE CARACTERES ---
def segmentar_caracteres(recorte_gris, grupo_ajustado):
    # Binarización robusta
    recorte_bin = cv2.adaptiveThreshold(
        recorte_gris, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 10
    )
    grupo_ordenado = sorted(grupo_ajustado, key=lambda r: r[0])
    caracteres = []
    img_marcados = cv2.cvtColor(recorte_gris, cv2.COLOR_GRAY2BGR)

    for (x, y, w, h) in grupo_ordenado:
        char_crop = recorte_bin[y:y+h, x:x+w]
        caracteres.append(char_crop)

        cv2.rectangle(img_marcados, (x, y), (x+w, y+h), (0, 255, 0), 1)

    return caracteres, img_marcados, recorte_bin



# --- FIGURA 4 ---




# --- MOSTRAR CARACTERES INDIVIDUALES (EJEMPLO) ---
nombre, _, _, _, grupo_adj, recorte_gris, estado = datos_graficos[3]

if estado.startswith("Detectado"):
    caracteres, img_marcados, recorte_bin = segmentar_caracteres(recorte_gris, grupo_adj)

    plt.figure(figsize=(12, 3))
    for i, char in enumerate(caracteres):
        plt.subplot(1, len(caracteres), i+1)
        plt.imshow(char, cmap='gray')
        plt.title(f"Char {i+1}")
        plt.axis('off')
    plt.show()



# --- FIGURA ÚNICA: TODAS LAS PATENTES EN 3 COLUMNAS × 4 FILAS ---

rows, cols = 4, 3
fig, axes = plt.subplots(rows, cols, figsize=(22, 18))
axes = axes.flatten()

for idx, ax in enumerate(axes):
    if idx >= len(datos_graficos):
        ax.axis('off')
        continue

    nombre, _, _, _, grupo_adj, recorte_gris, estado = datos_graficos[idx]

    if not estado.startswith("Detectado"):
        ax.imshow(np.zeros((80, 250)), cmap='gray')
        ax.set_title(f"{nombre}\nSin patente")
        ax.axis('off')
        continue

    # Segmentación
    caracteres, img_marcados, recorte_bin = segmentar_caracteres(recorte_gris, grupo_adj)

    # ------- Unificar los caracteres en una imagen horizontal -------
    espacios = 12  # espacio entre caracteres

    # Altura normalizada
    target_h = max([c.shape[0] for c in caracteres])
    caracteres_resized = []
    for c in caracteres:
        h, w = c.shape
        scale = target_h / h
        nw = int(w * scale)
        resized = cv2.resize(c, (nw, target_h), interpolation=cv2.INTER_NEAREST)
        caracteres_resized.append(resized)

    # Unir
    separador = 255 * np.ones((target_h, espacios), dtype=np.uint8)
    fila = caracteres_resized[0]
    for c in caracteres_resized[1:]:
        fila = np.hstack((fila, separador, c))

    # Mostrar en su celda del grid
    ax.imshow(fila, cmap='gray')
    ax.set_title(f"{nombre} — {len(caracteres)} chars")
    ax.axis('off')

plt.tight_layout()
plt.show()