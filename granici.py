import cv2
def detect_plate_number(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    plate_contour = None
    
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        area = cv2.contourArea(contour)
        if area < 10000:  # Настройка минимальной и максимальной площади
            plate_contour = contour
            break
        if len(approx) == 4:
            plate_contour = approx
            break
    if plate_contour is not None:

        x, y, w, h = cv2.boundingRect(plate_contour)
        plate_image = gray[y:y + h, x:x + w]
    return(plate_image)
image_path = "fullnum/1.jpg" 
plate_number = detect_plate_number(image_path)
cv2.imshow("Бинарное изображение", plate_number)
cv2.waitKey(1000)