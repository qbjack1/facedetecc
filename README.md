# Interactive Facial Detection Streamlit Web App

A lightweight computer vision portfolio project that detects faces in uploaded images using MediaPipe's TFLite face detector and displays the results in a Streamlit interface.

The app is designed to be compute-friendly: it runs inference on a single uploaded image, reports the number of detected faces, shows detection latency, and draws bounding boxes with confidence scores.

## Features

- Upload JPG, JPEG, PNG, or WEBP images.
- Adjust the minimum detection confidence from the sidebar.
- Run MediaPipe face detection with a bundled `detector.tflite` model.
- Compare the original image and detected-face output side by side.
- View face count and detection time metrics.

## Project Structure

```text
.
|-- app.py             # Streamlit app and MediaPipe detection pipeline
|-- detector.tflite    # Local face detector model used by MediaPipe
|-- face_1.jpg         # Sample image
|-- face_2.jpg         # Sample image
|-- face_3.png         # Sample image
|-- launch.bat         # Windows helper for launching the app
|-- requirements.txt   # Python dependencies
`-- README.md          # Project documentation
```

## Setup

This project was tested with Python 3.14.6.

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the app:

```powershell
python -m streamlit run app.py
```

On Windows, you can also run:

```powershell
.\launch.bat
```

## Usage

1. Open the Streamlit URL shown in the terminal.
2. Upload an image containing one or more faces.
3. Adjust the confidence threshold if needed.
4. Review the detected-face count, inference time, and bounding-box output.

## Notes

- This is a face detection project, not a face recognition or identity verification system.
- The model file `detector.tflite` must stay in the same folder as `app.py`.
- The included sample images can be used for quick manual testing.

## Possible Improvements

- Improve label styling for better readability on bright or busy images.
- Add a dropdown widget for the included sample images.
- Add an upscaling model for low-resolution images
- Add a downscaling model for high-resolution images
- Add webcam or video-frame support.
- Add downloadable result images.
- Add facial landmark features
- Add facial recognition features

## Limitations

- Because this project prioritizes lightweight, accessible computation, the face detector may be less robust on challenging images, such as low-light scenes, small faces, side profiles, occlusions, or crowded group photos.
