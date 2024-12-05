import cv2 as cv
import mediapipe as mp
import numpy as np
import csv
import copy
import itertools

print("Perform gestures and press 'a', 'b', 'c', 'd' for respective gestures.")
print("Press 'ESC' to quit.")

def main():
    # Camera preparation 
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 540)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6)

    while True:
        #Process KEY
        key = cv.waitKey(5)
        if key == 27:  # ESC
            break
        number = detect_number(key)
        ##Camera Capture
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                results.multi_handedness):
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                # Write to the dataset file
                logging_csv(number, pre_processed_landmark_list)

                # Drawing part
                mp_drawing = mp.solutions.drawing_utils
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(debug_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
        resized_debug_image = cv.resize(debug_image, None, fx=0.65, fy=0.65, interpolation=cv.INTER_AREA)
        cv.imshow('Hand Gesture Recognition', resized_debug_image)
    cap.release()
    cv.destroyAllWindows()


def logging_csv(number, landmark_list):
    if (0 <= number <= 9):
        csv_path = 'keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    return

def detect_number(key):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    return number

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

if __name__ == '__main__':
    main()