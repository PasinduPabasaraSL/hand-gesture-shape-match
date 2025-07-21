import cv2
import mediapipe as mp
import math
import numpy as np
import json
import os
import time

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

# Shape class
class Shape:
    def __init__(self, name, color, position):
        self.name = name
        self.color = color
        self.pos = position
        self.dragging = False
        self.matched = False
        self.drag_hold = 0

    def draw(self, img):
        if self.matched:
            return
        if self.name == "rectangle":
            cv2.rectangle(img, self.pos, (self.pos[0] + 60, self.pos[1] + 60), self.color, -1)
        elif self.name == "circle":
            cv2.circle(img, self.pos, 30, self.color, -1)
        elif self.name == "triangle":
            draw_triangle(img, self.pos, self.color)

# Draw triangle helper
def draw_triangle(img, center, color, filled=True):
    cx, cy = center
    pts = np.array([
        (cx, cy - 30),
        (cx - 30, cy + 30),
        (cx + 30, cy + 30)
    ], np.int32)
    if filled:
        cv2.fillPoly(img, [pts], color)
    else:
        cv2.polylines(img, [pts], True, color, 2)

# Point in triangle test (area method)
def point_in_triangle(pt, v1, v2, v3):
    def area(a, b, c):
        return abs((a[0]*(b[1]-c[1]) + b[0]*(c[1]-a[1]) + c[0]*(a[1]-b[1])) / 2.0)
    A = area(v1, v2, v3)
    A1 = area(pt, v2, v3)
    A2 = area(v1, pt, v3)
    A3 = area(v1, v2, pt)
    return abs((A1 + A2 + A3) - A) < 1e-3

# Check if point inside shape area
def is_in_shape_area(shape, px, py):
    x, y = shape.pos
    if shape.name == "rectangle":
        return x < px < x + 60 and y < py < y + 60
    elif shape.name == "circle":
        return math.hypot(px - x, py - y) < 30
    elif shape.name == "triangle":
        v1 = (x, y - 30)
        v2 = (x - 30, y + 30)
        v3 = (x + 30, y + 30)
        return point_in_triangle((px, py), v1, v2, v3)
    return False

# Check if inside bucket target
def is_inside_bucket(shape_name, px, py):
    bx, by = buckets[shape_name]
    return abs(bx - px) < 40 and abs(by - py) < 40

# Save progress
def save_progress(shapes):
    progress = {shape.name: shape.matched for shape in shapes}
    with open("progress.json", "w") as f:
        json.dump(progress, f)

# Load progress
def load_progress(shapes):
    if not os.path.exists("progress.json"):
        return
    with open("progress.json", "r") as f:
        data = json.load(f)
        for shape in shapes:
            if shape.name in data:
                shape.matched = data[shape.name]

# Detect thumbs up gesture to start game
def detect_thumbs_up(landmarks):
    thumb_up = landmarks[4].y < landmarks[3].y
    fingers_down = all(landmarks[i].y > landmarks[i - 2].y for i in [8, 12, 16, 20])
    return thumb_up and fingers_down

# Pinch detection between thumb and index finger (slightly relaxed threshold)
def is_pinching(landmarks):
    x1, y1 = landmarks[4].x, landmarks[4].y  # thumb tip
    x2, y2 = landmarks[8].x, landmarks[8].y  # index tip
    hand_size = math.hypot(landmarks[0].x - landmarks[9].x, landmarks[0].y - landmarks[9].y)
    distance = math.hypot(x2 - x1, y2 - y1)
    is_pinch = distance < 0.3 * hand_size  # relaxed from 0.25 to 0.3
    cx = int((x1 + x2) / 2 * 640)
    cy = int((y1 + y2) / 2 * 480)
    return is_pinch, cx, cy

# Initialize shapes and buckets
shapes = [
    Shape("rectangle", (0, 0, 255), (50, 100)),
    Shape("circle", (0, 255, 0), (50, 250)),
    Shape("triangle", (255, 0, 0), (50, 400))
]

buckets = {
    "rectangle": (500, 100),
    "circle": (500, 250),
    "triangle": (500, 400)
}

load_progress(shapes)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 640)
cap.set(4, 480)

selected_shape = None
game_started = False
start_time = 0
score = sum(1 for s in shapes if s.matched)
timer_duration = 60  # seconds

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    pinch = False
    pinch_x, pinch_y = 0, 0
    thumbs_up = False

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            pinch, pinch_x, pinch_y = is_pinching(handLms.landmark)
            if not game_started:
                thumbs_up = detect_thumbs_up(handLms.landmark)
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    # Show circle at pinch point for debugging
    if pinch:
        cv2.circle(img, (pinch_x, pinch_y), 10, (0, 255, 255), 3)

    if not game_started:
        cv2.putText(img, "Show Thumbs Up to Start", (120, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 3)
        if thumbs_up:
            game_started = True
            start_time = time.time()
        cv2.imshow("Shape Match Game", img)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    elapsed = int(time.time() - start_time)
    remaining = max(timer_duration - elapsed, 0)
    cv2.putText(img, f"Time Left: {remaining}s", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f"Score: {score}", (520, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if remaining == 0 or all(s.matched for s in shapes):
        save_progress(shapes)
        msg = "Game Over!" if remaining == 0 else "All Shapes Matched!"
        cv2.putText(img, msg, (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 255, 255), 3)
        cv2.imshow("Shape Match Game", img)
        if cv2.waitKey(0):
            break
        continue

    # Draw & update shapes
    for shape in shapes:
        shape.draw(img)

        if shape.matched:
            continue

        # Select shape for dragging
        if pinch and not selected_shape and is_in_shape_area(shape, pinch_x, pinch_y):
            shape.dragging = True
            selected_shape = shape
            shape.drag_hold = 3  # frames hold after pinch lost

        if shape.dragging:
            # Clamp position (accounting for 30 px radius)
            new_x = min(max(pinch_x, 30), 640 - 30)
            new_y = min(max(pinch_y, 30), 480 - 30)
            shape.pos = (new_x, new_y)

            if not pinch:
                shape.drag_hold -= 1
                if shape.drag_hold <= 0:
                    shape.dragging = False
                    selected_shape = None
                    if is_inside_bucket(shape.name, pinch_x, pinch_y):
                        if not shape.matched:
                            shape.matched = True
                            score += 1
                        save_progress(shapes)

    # Draw buckets
    for name, pos in buckets.items():
        if name == "rectangle":
            cv2.rectangle(img, pos, (pos[0] + 60, pos[1] + 60), (200, 200, 200), 2)
        elif name == "circle":
            cv2.circle(img, pos, 30, (200, 200, 200), 2)
        elif name == "triangle":
            draw_triangle(img, pos, (200, 200, 200), filled=False)

    cv2.imshow("Shape Match Game", img)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to quit
        break
    elif key == ord('r'):  # 'r' to reset progress
        for shape in shapes:
            shape.matched = False
        score = 0
        if os.path.exists("progress.json"):
            os.remove("progress.json")

cap.release()
cv2.destroyAllWindows()