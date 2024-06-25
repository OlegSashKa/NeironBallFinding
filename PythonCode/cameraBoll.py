from ultralytics import YOLO
import cv2

def PutText(image, informations, x, y):
    org = (x, y) 
    font = cv2.FONT_HERSHEY_COMPLEX # Шрифт текста
    font_scale = 0.7 # Размер шрифта
    color = (255, 0, 0) # Цвет текста (BGR)  # Синий цвет
    thickness = 2 # Толщина линий текста
    cv2.putText(image, informations, org, font, font_scale, (0,0,0), 3)
    cv2.putText(image, informations, org, font, font_scale, (255,0,0), 2)
    

model = YOLO(r'runs\detect\train4\weights\best.pt')

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920/2)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080/2)

while cap.isOpened:
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)

    # If frame is read correctly ret is True
    if ret:
        result = model(frame, conf=0.60)

        annotated_frame = result[0].plot()
        boxes = result[0].boxes

        for box in boxes:
            # Извлечение координат
            x1, y1, x2, y2 = box.xyxy[0]  # формат [x1, y1, x2, y2]

            # Вы можете также получить другие атрибуты, такие как:
            confidence = box.conf[0]  # уверенность
            class_id = box.cls[0]     # идентификатор класса
            info = f"{confidence}, {class_id}"
            PutText(annotated_frame, info, 50, 50)
            PutText(annotated_frame, f"{x1} {y1}", 50, 100)
        cv2.imshow("222", annotated_frame)

        if cv2.waitKey(1) == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
