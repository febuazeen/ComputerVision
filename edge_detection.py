import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. INPUT GAMBAR
# =========================
image = cv2.imread('G:/1. KULIYAH/semester6/Computer Vision/image10.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# =========================
# 2. GRAYSCALE
# =========================
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# =========================
# 3. GAUSSIAN BLUR
# =========================
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# =========================
# 4. EDGE FEATURE (CANNY)
# =========================
edges = cv2.Canny(blur, 50, 120)

kernel = np.ones((3,3), np.uint8)
edges = cv2.dilate(edges, kernel, iterations=1)

# =========================
# 5. CONTOUR FEATURE
# =========================
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

output = image_rgb.copy()

# =========================
# 6. FEATURE EXTRACTION + KLASIFIKASI
# =========================
for cnt in contours:

    # =========================
    # GEOMETRIC FEATURE
    # =========================
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    circularity = 4 * np.pi * area / (perimeter * perimeter)

    if area < 100:
        continue

    # =========================
    # SHAPE FEATURE (Approx)
    # =========================
    approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)

    # =========================
    # SPATIAL FEATURE (Bounding Box)
    # =========================
    x, y, w, h = cv2.boundingRect(approx)

    # =========================
    # KLASIFIKASI BENTUK
    # =========================
    if len(approx) == 3:
        shape = "Segitiga"

    elif len(approx) == 4:
        ratio = w / float(h)
        if 0.9 <= ratio <= 1.1:
            shape = "Persegi"
        else:
            shape = "Persegi Panjang"

    elif len(approx) > 4:
        if circularity > 0.8:
            shape = "Lingkaran"
        else:
            shape = "Tidak diketahui"

    # =========================
    # VISUALISASI
    # =========================
    info = f"{shape} | L:{int(area)} K:{int(perimeter)}"

    cv2.drawContours(output, [approx], -1, (0, 255, 0), 2)

    cv2.putText(output, info, (x, y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.putText(output, f"({x},{y})", (x, y+h+15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# =========================
# 7. VISUALISASI
# =========================
plt.figure(figsize=(12,10))

plt.subplot(2,2,1)
plt.title("Gambar Asli")
plt.imshow(image_rgb)
plt.axis('off')

plt.subplot(2,2,2)
plt.title("Grayscale")
plt.imshow(gray, cmap='gray')
plt.axis('off')

plt.subplot(2,2,3)
plt.title("Edge (Canny)")
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.subplot(2,2,4)
plt.title("Hasil Deteksi Bentuk")
plt.imshow(output)
plt.axis('off')

plt.tight_layout()
plt.show()
