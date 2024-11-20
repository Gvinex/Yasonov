import cv2
import numpy as np
from openpyxl import Workbook



camera_position = np.array([660.0, 760.0, 35.0])  # (x, y, z) камеры в метрах
camera_azimuth = -110.0  # азимут

def pixel_to_camera_coords(pixel_coords, frame_width, frame_height, camera_position, camera_azimuth):
    """
    """
    scale_factor = 500.0 / frame_width  # Примерный масштаб в метрах на пиксель

    # Преобразуем координаты пикселей в локальные метры
    local_x = (pixel_coords[0] - frame_width / 2) * scale_factor
    local_y = (pixel_coords[1] - frame_height / 2) * scale_factor

    # Учет азимута камеры (вращение вокруг оси Z)
    angle_rad = np.radians(camera_azimuth)
    rotated_x = local_x * np.cos(angle_rad) - local_y * np.sin(angle_rad)
    rotated_y = local_x * np.sin(angle_rad) + local_y * np.cos(angle_rad)

    # Координаты относительно камеры
    global_x = camera_position[0] + rotated_x
    global_y = camera_position[1] + rotated_y
    global_z = camera_position[2]

    return global_x, global_y, global_z

# Пути к видеофайлам
input_video_path = 'D:/Python Projects/Work/hackathon_with_peleng/videoset1/Seq1_camera1T.mov'
output_video_path = 'D:/Python Projects/Work/hackathon_with_peleng/videoset1/detected_moving_circles.avi'
output_excel_path = 'D:/Python Projects/Work/hackathon_with_peleng/videoset1/circle_coordinates.xlsx'


cap = cv2.VideoCapture(input_video_path)


if not cap.isOpened():
    print("Ошибка: не удалось открыть видео.")
    exit()


frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Инициализируем вычитание фона
back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)


wb = Workbook()
ws = wb.active
ws.title = "Circle Coordinates"
ws.append(["Frame", "X", "Y"])


save_frequency = int(fps * 0.5)
frame_count = 0

# Обрабатываем видео по кадрам
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Преобразуем кадр в градации серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Применяем вычитание фона
    fg_mask = back_sub.apply(gray)

    # Убираем шум с помощью морфологических операций
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    # Применяем размытие для улучшения контуров
    fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)

    # Ищем контуры в маске
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Пропускаем слишком маленькие области
        if cv2.contourArea(contour) < 300:  # Уменьшен порог
            continue

        # Попытка вписать окружность
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)

        if radius > 6:  # Уменьшен минимальный радиус
            # Проверка на округлость
            circularity = (4 * np.pi * cv2.contourArea(contour)) / (cv2.arcLength(contour, True) ** 2)
            if 0.85 < circularity <= 1.15:  # Более широкий диапазон для округлости
                # зеленый круг
                cv2.circle(frame, center, radius, (0, 255, 0), 2)
                cv2.circle(frame, center, 2, (0, 0, 255), 3)

                # Сохраняем координаты в Excel каждые 0.5 секунды
                if frame_count % save_frequency == 0:
                    global_coords = pixel_to_camera_coords(center, frame_width, frame_height, camera_position,
                                                           camera_azimuth)
                    ws.append([frame_count, global_coords[0], global_coords[1], global_coords[2]])

    # Увеличиваем счетчик кадров
    frame_count += 1

    # Записываем обработанный кадр в выходное видео
    out.write(frame)

# Сохраняем Excel-файл
wb.save(output_excel_path)

# Освобождаем ресурсы
cap.release()
out.release()

print(f"Обработанное видео сохранено в {output_video_path}")
print(f"Координаты окружностей сохранены в {output_excel_path}")

