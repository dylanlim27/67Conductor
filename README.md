# 67Conductor

67Conductor is an interactive, multi-disciplinary software system originally built for HackNotts. It combines **Computer Vision (MediaPipe & TensorFlow.js)**, **Haskell FFI & Subprocess computation**, and a **Node.js Express Server powered by LLM Text-Generation** to create a gesture-controlled clothing swiping and styling application.

---

## Subprojects Overview

### 1. Computer Vision (`CompVision`)
Captures coordinates of wrists, shoulders, and elbows from video inputs using **MediaPipe Pose Landmarker** and translates them into gestures:
- **`bodypose.py`**: Broadcasts selected landmark coordinate frames over UDP (port 5005) to a Haskell receiver. It also listens on port 5006 for trigger patterns to display overlays.
- **`fashiontinder.py`**: A gesture-controlled visualizer where right-hand and left-hand gestures play "like" and "dislike" sound indicators.
- **`haslib.hs`**: A complete, compiled Haskell UDP receiver that processes coordinates, tracks gesture sequences (`A` and `B`), performs regular expression matching (`^(?:AB){2,}(?:A)?$|^(?:BA){2,}(?:B)?$`), and sends triggers back to Python.

### 2. LLM Backend Server (`Server_src`)
An Express server running locally on port `8080` that uses **Hugging Face Transformers** with the **SmolLM2-1.7B-Instruct** model to serve clothing styling recommendations:
- **`/start`**: Accepts an array of user preferences/tags and returns a curated clothing item description.
- **`/like` & `/dislike`**: Takes context from the previous suggestion and prompts the LLM to generate subsequent recommendations.
- **`classify_data.mjs`**: Image-to-text processing draft code using FastVLM ONNX community models.

### 3. Astro Frontend Website (`frontend_website`)
A modern static web app built with Astro, Tailwind CSS, and TypeScript:
- **Style Finder (`/`)**: A glassmorphic dark-mode web page where users select tags (e.g., `#casual`, `#vintage`, `#minimalist`) and swipe (Like/Dislike) on AI-recommended apparel in real-time. Connects dynamically to the `Server_src` backend on port 8080.
- **Pose Detection (`/model`)**: A client-side TensorFlow.js page running MoveNet, BlazePose, and PoseNet for browser-based posture tracking.

### 4. Haskell Experiments (`haskell`)
A playground containing examples of Haskell and Python interoperability:
- **FFI (`HSLib.hs` & `test_call.py`)**: Demonstrates calling Haskell functions from Python via `ctypes` using a custom C wrapper (`StartEnd.c`) and DLL definition (`HSLib.def`) to handle automatic GHC runtime system (RTS) initialization inside `DllMain`.
- **Subprocess (`angle_calc.hs` & `generated.py`)**: Computes elbow joint angles in radians from points piped into stdin, returning outputs back to Python via stdout.

---

## Setup & Running Guide

### Prerequisites
- **Node.js** (v20+ recommended)
- **Python 3.11** (with virtual environment capability)
- **GHC (Glasgow Haskell Compiler)** installed and on your PATH (required for compiling Haskell utilities)

---

### Step 1: Start the Backend Server (`Server_src`)
1. Navigate to the directory and install dependencies:
   ```bash
   cd Server_src
   npm install
   ```
2. Start the Express server:
   ```bash
   node index.mjs
   ```
   *Note: On first run, it will automatically download the SmolLM2 model (approx. 3.4GB) from Hugging Face.*

---

### Step 2: Start the Frontend Website (`frontend_website`)
1. Navigate to the directory and install dependencies:
   ```bash
   cd frontend_website
   npm install
   ```
2. Start the development server:
   ```bash
   npm run dev
   ```
3. Open your browser to `http://localhost:4321` to access the interactive web app.

---

### Step 3: Run the Computer Vision Visualizer (`CompVision`)
1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the pose tracker (replace `<camera_index>` with `0` or `1`):
   ```bash
   cd CompVision
   python bodypose.py <camera_index>
   ```
3. Compile and run the Haskell UDP logic (if desired):
   ```bash
   ghc -O2 haslib.hs
   ./haslib
   ```

---

### Step 4: Run the FFI / Subprocess Examples (`haskell`)
1. To run the FFI ctypes load test:
   ```bash
   cd haskell
   python test_call.py
   ```
2. To run the subprocess coordinate angle calculator:
   ```bash
   cd haskell
   python generated.py
   ```
