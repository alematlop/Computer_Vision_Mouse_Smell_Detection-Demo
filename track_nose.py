import cv2

# Load the video
video_path = 'mouse_video.mp4'  # replace with your video path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")

# Create a background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

# Manually define the object region (x, y, width, height)
object_roi = (100, 200, 50, 50)  # Replace with actual coordinates

proximity_threshold = 50  # Proximity threshold for smelling

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit if the video ends

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Find contours in the binary image
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through contours
    i = -1
    for contour in contours:
        i += 1
        if cv2.contourArea(contour) > 200:  # Ignore small movements (noise)
            # Get bounding box for the contour (mouse)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw a rectangle around the mouse

            # Track the center of the bounding box (approximation of the nose)
            mouse_nose_x = x + w // 2  # Center X of the bounding box
            mouse_nose_y = y + h // 2  # Center Y of the bounding box

            if i != 0:

                xp, yp, wp, hp = cv2.boundingRect(contours[-1])

                if yp > y: # mouse going downwards
                    if w > h:
                        mouse_nose_x = x
                        mouse_nose_y = y + h // 2
                    else:
                        mouse_nose_x = x + w // 2
                        mouse_nose_y = y




                # Draw a small circle to represent the mouse's nose
            cv2.circle(frame, (mouse_nose_x, mouse_nose_y), 5, (0, 0, 255), -1)  # Red dot for the nose

            # Check if the mouse is near the object (within the defined region)
            if (object_roi[0] < mouse_nose_x < object_roi[0] + object_roi[2] and
                    object_roi[1] < mouse_nose_y < object_roi[1] + object_roi[3]):
                # Calculate distance between the mouse's nose and the object's center
                object_center_x = object_roi[0] + object_roi[2] // 2
                object_center_y = object_roi[1] + object_roi[3] // 2
                distance = ((mouse_nose_x - object_center_x) ** 2 + (mouse_nose_y - object_center_y) ** 2) ** 0.5

                # If within proximity, count it as "smelling"
                if distance < proximity_threshold:
                    print("Mouse is smelling the object!")

    # Display the video frame
    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
