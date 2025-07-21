# Shape Match Game

An interactive computer vision game using **MediaPipe** and **OpenCV** where you use hand pinch gestures to drag and drop shapes into matching buckets.  
The game features multiple levels, a timer, scoring system, and progress saving.

## Features

- Hand gesture detection with MediaPipe  
- Pinch and drag shapes using your fingers  
- Multiple levels with different shapes  
- Timer and score tracking  
- Progress saving to JSON files  

## Installation

Make sure you have Python 3 installed. Then install required packages:

pip install opencv-python mediapipe numpy

## Usage
Run the game script:

```bash
python shape_match_game.py
```

Use a webcam and show a thumbs up gesture to start.

Pinch your thumb and index finger to select and drag shapes.

Drag shapes to their matching bucket to score points.

## Controls
Thumbs Up: Start the game

Pinch (thumb + index finger): Select and drag shapes

Escape (Esc) key: Quit the game
