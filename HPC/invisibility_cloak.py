import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# Allow the camera to warm up
for i in range(30):
    ret, background = cap.read()

# Flip the background horizontally
background = np.flip(background, axis=1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the frame horizontally
    frame = np.flip(frame, axis=1)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range of green color in HSV
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Perform morphological operations to get a cleaner mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((5, 5), np.uint8))

    # Create an inverse mask
    inv_mask = cv2.bitwise_not(mask)

    # Replace the green parts of the frame with background
    res1 = cv2.bitwise_and(frame, frame, mask=inv_mask)
    res2 = cv2.bitwise_and(background, background, mask=mask)

    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

    # Display the output
    cv2.imshow("Invisibility Cloak", final_output)

    # Press 'q' to exit loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
