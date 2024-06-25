import cv2
import time
 
def capture_frames(output_path):
    # Инициализация камеры
    cap = cv2.VideoCapture(0)
    width = 1280  # Новая ширина кадра
    height = 720  # Новая высота кадра
    # Установка размера захвата кадра
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print("Не удалось открыть камеру")
        return

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            # Захват кадра
            ret, frame = cap.read()
            if not ret:
                print("Не удалось захватить кадр")
                break

            # Получение текущего времени
            current_time = time.time()

            # Проверка, прошла ли 1 секунда с последнего сохранения кадра
            if current_time - start_time >= 1.0:
                # Сохранение кадра
                frame_path = f"{output_path}\{frame_count}.jpg"
                cv2.imwrite(frame_path, frame)
                print(f"Сохранен кадр: {frame_path}")

                # Увеличение счетчика кадров
                frame_count += 1

                # Обновление времени последнего сохранения
                start_time = current_time
            cv2.imshow("VideoForTraining", frame)
            if cv2.waitKey(1) == 27:
                break

    except KeyboardInterrupt:
        print("Остановка захвата кадров")

    finally:
        # Освобождение ресурса камеры
        cap.release()
        cv2.destroyAllWindows()

# Укажите путь, куда будут сохраняться кадры
output_path = r"RedBall\train\image2"

# Вызов функции захвата кадров
capture_frames(output_path)