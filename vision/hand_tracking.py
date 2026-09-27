import cv2
import mediapipe as mp
import numpy as np


class HandTracker:

    def __init__(self):

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

    def process(self, frame):

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        return self.hands.process(rgb)

    def draw_landmarks(self, frame, result):

        if result.multi_hand_landmarks:

            for hand_landmarks in result.multi_hand_landmarks:

                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

    def get_gesture(self, result):

        if not result.multi_hand_landmarks:
            return "none"

        hand = result.multi_hand_landmarks[0].landmark

        # Finger states
        index_up = hand[8].y < hand[6].y
        middle_up = hand[12].y < hand[10].y
        ring_up = hand[16].y < hand[14].y
        pinky_up = hand[20].y < hand[18].y

        # Thumb
        thumb_open = hand[4].x < hand[3].x

        # Pinch distance
        thumb = np.array([hand[4].x, hand[4].y])
        index = np.array([hand[8].x, hand[8].y])

        pinch_distance = np.linalg.norm(
            thumb - index
        )

        if pinch_distance < 0.055:
            return "pinch"

        # Fist
        if (
            not index_up
            and not middle_up
            and not ring_up
            and not pinky_up
        ):
            return "cloud"

        # Two fingers
        if (
            index_up
            and middle_up
            and not ring_up
            and not pinky_up
        ):
            return "heart"

        # One finger
        if (
            index_up
            and not middle_up
            and not ring_up
            and not pinky_up
        ):
            return "star"

        # Open palm
        if (
            index_up
            and middle_up
            and ring_up
            and pinky_up
        ):
            return "galaxy"

        return "galaxy"

    def get_hand_position(self, result):

        if not result.multi_hand_landmarks:
            return None

        hand = result.multi_hand_landmarks[0].landmark

        x = hand[8].x
        y = hand[8].y

        return x, y

    def get_pinch_strength(self, result):

        if not result.multi_hand_landmarks:
            return 1.0

        hand = result.multi_hand_landmarks[0].landmark

        thumb = np.array(
            [hand[4].x, hand[4].y]
        )

        index = np.array(
            [hand[8].x, hand[8].y]
        )

        distance = np.linalg.norm(
            thumb - index
        )

        scale = np.interp(
            distance,
            [0.03, 0.25],
            [1.8, 0.75]
        )

        return float(
            np.clip(scale, 0.7, 1.8)
        )

    def release(self):

        self.hands.close()