import cv2
import numpy as np
import streamlit as st
import numpy as np
from io import BytesIO


cam_id = 1


def detect_color(frame, lower_bound, upper_bound, color_name):
    # Converter a imagem para o espaço de cores HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Criar a máscara para a cor
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Encontrar contornos na máscara
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Desenhar retângulos ao redor dos contornos detectados
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filtrar pequenos contornos
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                color_name,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

    return frame


# Initialize the webcam capture and store it in session state
if "cap" not in st.session_state or st.session_state.cap is None:
    print("Initializing webcam")
    st.session_state.cap = cv2.VideoCapture(cam_id)

st.title("Detecção de Cores")
st.write("mexe nos slider ai vai")


def generate_frame():
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


slider_red = st.slider("Red", 0, 255, value=[50, 100], key="red")
slider_green = st.slider("Green", 0, 255, value=[75, 125], key="green")
slider_blue = st.slider("Blue", 0, 255, value=[120, 220], key="blue")

filter_min = np.array([100, 0, 0])
filter_max = np.array([255, 255, 255])

frame_placeholder = st.empty()


def capture_sliders():
    filter_min = np.array([slider_red[0], slider_green[0], slider_blue[0]])
    filter_max = np.array([slider_red[1], slider_green[1], slider_blue[1]])
    return filter_min, filter_max


while True:
    filter_min, filter_max = capture_sliders()

    ret, frame = st.session_state.cap.read()
    if not ret:
        break

    print("filter_min", filter_min)
    print("filter_max", filter_max)
    frame = detect_color(frame, filter_min, filter_max, "Object")

    frame_placeholder.image(frame, channels="BGR")


st.session_state.cap.release()
cv2.destroyAllWindows()
