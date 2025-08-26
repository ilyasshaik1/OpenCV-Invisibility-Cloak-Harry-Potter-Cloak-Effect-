
# 🧙 OpenCV Invisibility Cloak (Harry Potter Cloak Effect)

This project recreates the famous **Harry Potter invisibility cloak** effect using OpenCV in Python.  
It works in real-time with your webcam by replacing a cloak of a specific color (red, blue, or green) with the background.

---

## ✨ Features
- Real-time cloak invisibility effect
- Supports **Red, Blue, Green** cloaks
- HSV trackbar calibration for tricky lighting
- FPS counter
- Snapshot saving (`./frames/frame_<timestamp>.png`)
- Keyboard controls

---

## ⚙️ Requirements
- Python 3.8+
- [PyCharm IDE](https://www.jetbrains.com/pycharm/) (recommended)

### Install required libraries:
```bash
pip install opencv-python numpy
````

---

## ▶️ How to Run

1. Clone/download this project.
2. Open it in **PyCharm**.
3. Install dependencies (see above).
4. Run the script:

   ```bash
   python invisibility_cloak.py
   ```

---

## 🎮 Controls

* **b** → Capture background (stand out of the frame, keep scene still)
* **1 / 2 / 3** → Switch target cloak color (Red / Blue / Green)
* **c** → Toggle HSV calibration sliders
* **s** → Save snapshot (`./frames/frame_<timestamp>.png`)
* **q** → Quit program

---

## 📷 Tips for Best Results

* Use a **solid bright cloak** (preferably red, blue, or green).
* Ensure **good lighting** with minimal shadows.
* Keep the background still when capturing.
* If edges flicker, press **c** and adjust HSV sliders.

---

## 📝 License

This project is for **educational purposes only**.
You’re free to use and modify it for learning.

