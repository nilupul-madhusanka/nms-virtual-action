import cv2
import numpy as np
import mediapipe as mp
import pygame

# Initialize mediapipe hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize pygame mixer
pygame.mixer.init()
pygame.mixer.music.load('song.mp3')

# Initialize webcam
cap = cv2.VideoCapture(0)

# Button parameters
button_x, button_y, button_w, button_h = 200, 100, 200, 100
song_playing = False

def draw_button(img, text, x, y, w, h):
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, text, (x + 20, y + 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

def is_finger_in_button(x, y, bx, by, bw, bh):
    return bx < x < bx + bw and by < y < by + bh

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame to detect hands
    result = hands.process(frame_rgb)
    draw_button(frame, "PLAY", button_x, button_y, button_w, button_h)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the landmarks for the index finger tip and thumb tip
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            # Convert normalized coordinates to pixel values
            index_finger_x, index_finger_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)

            # Draw a circle on the index finger tip
            cv2.circle(frame, (index_finger_x, index_finger_y), 10, (0, 255, 0), cv2.FILLED)

            # Calculate the distance between index finger tip and thumb tip
            distance = np.linalg.norm(np.array([index_finger_x, index_finger_y]) - np.array([thumb_x, thumb_y]))

            # If distance is small, consider it as a click
            if distance < 30 and is_finger_in_button(index_finger_x, index_finger_y, button_x, button_y, button_w, button_h):
                if not song_playing:
                    pygame.mixer.music.play()
                    song_playing = True
                    print("Playing song...")

    # Display the frame
    cv2.imshow("Virtual Button Click", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Check if the song is still playing
    if not pygame.mixer.music.get_busy() and song_playing:
        song_playing = False
        print("Song finished playing.")

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()
pygame.mixer.quit()
