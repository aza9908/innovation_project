import cv2
cap = cv2.VideoCapture(0)

if not cap.isOpened():
	print("Camera not opened")
	exit()
	
ret, frame = cap.read()

if ret:
	cv2.imwrite("test.jpg", frame)
	print("Image saved as test.jpg")
else:
	print("Failed to capture")
	
cap.release()
