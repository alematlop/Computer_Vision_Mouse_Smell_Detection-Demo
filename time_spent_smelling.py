import cv2
import math

# Constants
SMELLING_POINT = (155, 175)  # Coordinates of the object being smelled
SMELLING_RADIUS = 20  # Radius for smelling detection
MIN_DISTANCE = 20  # Minimum distance for valid nose movement
MAX_DISTANCE = 40  # Maximum distance for valid nose movement
MIN_AREA = 200  # Minimum contour area to consider
MAX_AREA = 500  # Maximum contour area to consider

def distance_between(p1, p2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def detect_nose(contour):
    """Detect the nose point from a contour."""
    M = cv2.moments(contour)
    if M['m00'] == 0:
        return None

    # Calculate centroid
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    # Find the farthest point from the centroid (nose)
    max_distance = 0
    nose_point = None
    for point in contour:
        x, y = point[0]
        distance = (x - cx) ** 2 + (y - cy) ** 2
        if distance > max_distance:
            max_distance = distance
            nose_point = (x, y)

    return nose_point

def main():
    # Load video
    cap = cv2.VideoCapture('mouse_video_cropped.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    nose_history = []
    smelling_frames = 0

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale and threshold
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

        # Noise reduction
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by area
        filtered_contours = [cnt for cnt in contours if MIN_AREA < cv2.contourArea(cnt) < MAX_AREA]

        # Process each contour
        for contour in filtered_contours:
            # Detect the nose point
            nose_point = detect_nose(contour)
            if nose_point:
                # Smooth the nose position
                dist = 0
                if nose_history:
                    dist = distance_between(nose_point, nose_history[-1])

                if dist < MIN_DISTANCE or dist > MAX_DISTANCE:
                    nose_history.append(nose_point)
                else:
                    nose_history.append(nose_history[-1])

                # Check if the mouse is smelling
                if distance_between(nose_history[-1], SMELLING_POINT) < SMELLING_RADIUS:
                    smelling_frames += 1

        # Print progress
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames ({100 * frame_idx / total_frames:.1f}%)")

    # Release resources
    cap.release()

    # Calculate the time spent smelling
    time_spent_smelling = smelling_frames / fps
    print(f"Time Spent Smelling: {time_spent_smelling:.2f} seconds")

if __name__ == "__main__":
    main()