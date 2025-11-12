import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---  PREPROCESAMIENTO ---
img = cv2.imread("monedas.jpg", cv2.IMREAD_GRAYSCALE)
# Para HoughCircles, un desenfoque de mediana suele ser muy bueno para preservar los bordes.
blur = cv2.medianBlur(img, 9)

# ---  DETECCIÓN DE CÍRCULOS CON HOUGH ---
#   - dp: Relación inversa de resolución. Siempre 1.
#   - minDist: Distancia mínima entre centros de círculos detectados.
#   - param1: Umbral superior para el detector Canny interno de Hough.
#   - param2: Umbral de "votos". Cuanto más bajo, más círculos (incluso falsos) detectará.
#   - minRadius/maxRadius: Rango de radios de los círculos a buscar.
circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                           param1=150, param2=40,
                           minRadius=80, maxRadius=250)

# ---  VISUALIZACIÓN ---
img_with_circles = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Asegurarse de que se encontraron círculos antes de procesarlos
if circles is not None:
    circles = np.uint16(np.around(circles))
    print(f"Se encontraron {len(circles[0, :])} monedas con HoughCircles.")
    
    # Dibujar los círculos encontrados
    for i in circles[0, :]:
        # Dibujar el contorno del círculo
        cv2.circle(img_with_circles, (i[0], i[1]), i[2], (0, 255, 0), 3)
        # Dibujar el centro del círculo
        cv2.circle(img_with_circles, (i[0], i[1]), 2, (0, 0, 255), 3)
else:
    print("No se encontraron monedas con los parámetros actuales.")

# Mostrar el resultado
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(img_with_circles, cv2.COLOR_BGR2RGB))
plt.title(f'Detección con HoughCircles')
plt.axis('off')
plt.show()

if circles is not None:
    circles = np.uint16(np.around(circles))
    radios = circles[0, :, 2]
    #print(f"Se detectaron {len(radios)} monedas.")

    # --- CLASIFICACIÓN POR TAMAÑO ---
    # Ajustá estos valores según tu imagen (se calculan en píxeles)
    # Podés imprimir np.sort(radios) para ver los valores detectados y afinar los límites.
    small_limit = 145   # límite superior para monedas pequeñas (10 cent)
    large_limit = 170  # límite superior para monedas medianas (1 peso)


    categorias = {
        "Pequeñas (10¢)": [],
        "Medianas (1 peso)": [],
        "Grandes (50¢)": []
    }

    for i in circles[0, :]:
        r = i[2]
        if r < small_limit:
            categorias["Pequeñas (10¢)"].append(i)
        elif  small_limit < r < large_limit:
            categorias["Medianas (1 peso)"].append(i)
        else:
            categorias["Grandes (50¢)"].append(i)

    # --- RESULTADOS ---
    print("\nClasificación de monedas por tamaño:")
    for tipo, lista in categorias.items():
        print(f"{tipo}: {len(lista)} monedas")

    # --- VISUALIZACIÓN CON COLORES POR CLASE ---
    colores = {
        "Pequeñas (10¢)": (255, 0, 0),    # Azul
        "Medianas (1 peso)": (0, 255, 0),    # Verde
        "Grandes (50¢)": (0, 0, 255)         # Rojo
    }

    img_clasif = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for tipo, lista in categorias.items():
        for i in lista:
            cv2.circle(img_clasif, (i[0], i[1]), i[2], colores[tipo], 3)
            cv2.circle(img_clasif, (i[0], i[1]), 2, (0, 0, 0), 2)

    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(img_clasif, cv2.COLOR_BGR2RGB))
    plt.title('Clasificación de monedas según tamaño (ajustada)')
    plt.axis('off')
    plt.show()

else:
    print("No se detectaron monedas con los parámetros actuales.")

# Umbral (170 es un buen valor)
thresh_value = 170
_, mask_dados = cv2.threshold(img, thresh_value, 255, cv2.THRESH_BINARY)

# Buscar contornos
contornos, _ = cv2.findContours(mask_dados, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# ---- PARÁMETROS DE FILTRADO (AJUSTADOS) ----
min_area_cara = 35000      # Un área mínima más segura que 500
epsilon_perc = 0.04       # CAMBIO 1: 4% (más flexible que 0.02)
min_aspect = 0.7          # CAMBIO 2: Rango más amplio
max_aspect = 1.3          # CAMBIO 2: Rango más amplio

print("--- Depurando Contornos (Área > 1500) ---")

# Iterar sobre los contornos
for cnt in contornos:
    area = cv2.contourArea(cnt)
    
    # Filtro de área (para ruido)
    if area > min_area_cara:
        
        # CAMBIO 1 (en la fórmula): Usar epsilon_perc
        approx = cv2.approxPolyDP(cnt, epsilon_perc * cv2.arcLength(cnt, True), True)
        num_vertices = len(approx)
        
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = 1.0 # Valor default
        if h > 0: # Evitar división por cero
            aspect_ratio = float(w) / h
            
        # Imprimimos info de CADA contorno interesante para depurar
        print(f"Contorno. Área: {area:.0f}, Vértices: {num_vertices}, Aspect: {aspect_ratio:.2f}")

        # Si tiene 4 lados (es un cuadrilátero)
        if num_vertices == 4:
            
            # CAMBIO 2 (en los límites): Usar el rango más flexible
            if min_aspect <= aspect_ratio <= max_aspect:
                print("  -> ¡DADO DETECTADO!")
                cv2.drawContours(img_color, [approx], 0, (0, 0, 255), 4)
            else:
                print("  -> Descartado (Aspect Ratio fuera de rango)")
        else:
            print("  -> Descartado (No tiene 4 vértices)")

# (Opcional) Habría que agregar el filtro de "contar una sola vez"
# que teníamos antes, para que si un dado muestra 2 caras, solo cuente 1.
# Pero este código debería *dibujar* las caras de ambos dados.

plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
plt.title("Detección con Filtros Flexibles")
plt.show()
















































