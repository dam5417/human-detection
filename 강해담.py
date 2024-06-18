import numpy as np
import cv2

brightness = 100  # 초기 밝기 설정
frame = None  # 전역 변수로 frame 선언
rect_start = (0, 0)
rect_end = (0, 0)
is_drawing = False

def adjust_brightness(value):
    global brightness, frame
    brightness = value - 100  # 트랙바 값에서 100을 뺀 만큼 밝기 조절
    if frame is not None:
        adjusted_img = np.clip(frame.astype(np.int16) + brightness, 0, 255).astype(np.uint8)
        cv2.imshow("Object detection", adjusted_img)

def apply_laplacian(img, start, end):
    x1, y1 = min(start[0], end[0]), min(start[1], end[1])
    x2, y2 = max(start[0], end[0]), max(start[1], end[1])
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, img.shape[1]), min(y2, img.shape[0])

    if x1 >= x2 or y1 >= y2:
        return

    roi = img[y1:y2, x1:x2]  # 선택된 영역 추출
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # 흑백 변환
    laplacian = cv2.Laplacian(gray_roi, cv2.CV_64F)  # 라플라시안 에지 검출
    laplacian = np.uint8(np.absolute(laplacian))  # 부호 없는 정수로 변환
    img[y1:y2, x1:x2] = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)  # 컬러로 변환하여 적용
    cv2.imshow("Object detection", img)  # 에지 검출 결과를 표시

def onMouse(event, x, y, flags, param):
    global rect_start, rect_end, is_drawing, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        rect_start = (x, y)
        is_drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if is_drawing:
            rect_end = (x, y)
            frame_copy = np.copy(frame)
            cv2.rectangle(frame_copy, rect_start, rect_end, (0, 255, 0), 2)
            cv2.imshow("Object detection", frame_copy)
    elif event == cv2.EVENT_LBUTTONUP:
        is_drawing = False
        rect_end = (x, y)
        apply_laplacian(frame, rect_start, rect_end)
        cv2.waitKey(0)

def process_video():
    global frame
    video = cv2.VideoCapture(0)  # 카메라 입력으로 변경
    while video.isOpened():
        success, frame = video.read()
        if success:
            height, width, _ = frame.shape
            blob = cv2.dnn.blobFromImage(frame, 1.0 / 256, (448, 448), (0, 0, 0), swapRB=True, crop=False)
            yolo_model.setInput(blob)
            output3 = yolo_model.forward(out_layers)
            class_ids, confidences, boxes = [], [], []
            for output in output3:
                for vec85 in output:
                    scores = vec85[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        centerx, centery = int(vec85[0] * width), int(vec85[1] * height)
                        w, h = int(vec85[2] * width), int(vec85[3] * height)
                        x, y = int(centerx - w / 2), int(centery - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    text = str(classes[class_ids[i]]) + '%.3f' % confidences[i]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), colors[class_ids[i]], 2)
                    cv2.putText(frame, text, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 2, colors[class_ids[i]], 2)

            if 0 in class_ids:
                print('사람이 검출됨!!!')

            cv2.imshow("Object detection", frame)
            edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 100, 200)
            cv2.imshow("Canny Edges", edges)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

            if 0 in class_ids:

                roi = frame[y:y + h, x:x + w]

                flipped_roi1 = cv2.flip(roi, 0)
                blue_channel = flipped_roi1[:, :, 0]

                flipped_roi2 = cv2.flip(roi, 1)
                green_channel = flipped_roi2[:, :, 1]

                flipped_roi3 = cv2.flip(roi, -1)
                red_channel = flipped_roi3[:, :, 2]

                cv2.imshow('Blue Channel', blue_channel)
                cv2.imshow('Green Channel', green_channel)
                cv2.imshow('Red Channel', red_channel)

    video.release()
    cv2.destroyAllWindows()

# 객체 검출을 위한 YOLO 모델 로딩
classes = []
with open('coco.names.txt', 'r') as f:
    classes = [line.strip() for line in f.readlines()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
yolo_model = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
layer_names = yolo_model.getLayerNames()
out_layers = [layer_names[i - 1] for i in yolo_model.getUnconnectedOutLayers()]

# 초기화할 때 트랙바 생성
cv2.namedWindow("Object detection")
cv2.createTrackbar("Brightness", "Object detection", 100, 200, adjust_brightness)

# 마우스 이벤트 콜백 등록
cv2.setMouseCallback("Object detection", onMouse)

# 영상 처리 및 객체 검출 시작
process_video()
