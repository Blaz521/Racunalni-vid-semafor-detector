import cv2
import numpy as np

cap = cv2.VideoCapture("DayDrive2.mp4")

while True:
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key == ord('d'):
        for _ in range(10):
            if not cap.grab():
                break
    ret, frame = cap.read()
    if not ret:
        break

    height = frame.shape[0]
    frame = frame[:height // 2, :]

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    red_lower = np.array([0, 150, 150], np.uint8)
    red_upper = np.array([10, 255, 255], np.uint8)

    red_lower2 = np.array([170, 150, 150], np.uint8)
    red_upper2 = np.array([180, 255, 255], np.uint8)

    yellow_lower = np.array([20, 150, 200], np.uint8)
    yellow_upper = np.array([35, 255, 255], np.uint8)

    green_lower = np.array([86, 30, 140], np.uint8)
    green_upper = np.array([95, 255, 255], np.uint8)

    red_mask1 = cv2.inRange(hsv_frame, red_lower, red_upper)
    red_mask2 = cv2.inRange(hsv_frame, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    yellow_mask = cv2.inRange(hsv_frame, yellow_lower, yellow_upper)
    green_mask = cv2.inRange(hsv_frame, green_lower, green_upper)

    ryg = cv2.bitwise_or(red_mask, yellow_mask)
    ryg = cv2.bitwise_or(ryg, green_mask)

    GausBlur = cv2.GaussianBlur(ryg, (7, 7), 0)
    cv2.imshow("gaus", GausBlur)
    contours, _ = cv2.findContours(GausBlur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        detectionArea = cv2.contourArea(contour)
        if 100 < detectionArea < 3000:
            x, y, w, h = cv2.boundingRect(contour)
            ratio = w / float(h)
            if 0.75 <= ratio <= 1.25:
                roi = frame[y:y+h, x:x+w]
                if roi.size == 0:
                    continue

                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                gray_roi = cv2.medianBlur(gray_roi, 5)

                circles = cv2.HoughCircles(
                    gray_roi,
                    cv2.HOUGH_GRADIENT,
                    dp=1,
                    minDist=min(w, h) // 3,
                    param1=50,
                    param2=15,
                    minRadius=int(min(w, h) * 0.3),
                    maxRadius=int(max(w, h) * 0.6)
                )
                if circles is not None:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    cv2.putText(frame, "Semafor", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.imshow("Detekcija semafora", frame)

cap.release()
cv2.destroyAllWindows()
