import streamlit as st
import torch
import numpy as np
import cv2
import av

from PIL import Image
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase
)

from unet import UNet


# =====================================
# COLOR MAP
# =====================================

COLORS = np.array([
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70]
], dtype=np.uint8)


# =====================================
# DEVICE
# =====================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# =====================================
# MODEL
# =====================================

@st.cache_resource
def load_model():

    model = UNet()

    model.load_state_dict(
        torch.load(
            "best_unet_camvid.pth",
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    return model


model = load_model()

st.success("✅ Model Loaded Successfully")
st.write("Device:", device)


# =====================================
# PREDICTION FUNCTION
# =====================================

def predict_segmentation(frame):

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    img = cv2.resize(
        rgb,
        (256, 256)
    )

    img = img.astype(np.float32) / 255.0

    # Uncomment if you used ImageNet normalization
    """
    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32
    )

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32
    )

    img = (img - mean) / std
    """

    tensor = torch.tensor(
        img.transpose(2, 0, 1),
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(tensor)

        print("Output Shape:", output.shape)

        pred = torch.argmax(
            output,
            dim=1
        ).squeeze().cpu().numpy()

        print("Min Class:", pred.min())
        print("Max Class:", pred.max())

    pred = np.clip(
        pred,
        0,
        len(COLORS) - 1
    )

    colored_mask = COLORS[pred]

    colored_mask = cv2.resize(
        colored_mask,
        (
            frame.shape[1],
            frame.shape[0]
        ),
        interpolation=cv2.INTER_NEAREST
    )

    overlay = cv2.addWeighted(
        frame,
        0.6,
        colored_mask,
        0.4,
        0
    )

    return colored_mask, overlay


# =====================================
# WEBCAM PROCESSOR
# =====================================

class VideoProcessor(VideoProcessorBase):

    def recv(self, frame):

        try:

            print("Frame Received")

            img = frame.to_ndarray(
                format="bgr24"
            )

            _, overlay = predict_segmentation(img)

            return av.VideoFrame.from_ndarray(
                overlay,
                format="bgr24"
            )

        except Exception as e:

            print("WEBCAM ERROR:", e)

            img = frame.to_ndarray(
                format="bgr24"
            )

            return av.VideoFrame.from_ndarray(
                img,
                format="bgr24"
            )


# =====================================
# UI
# =====================================

st.title("🚗 CamVid Semantic Segmentation using U-Net")

mode = st.radio(
    "Choose Input",
    ["Image Upload", "Webcam"]
)


# =====================================
# IMAGE MODE
# =====================================

if mode == "Image Upload":

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        image_np = np.array(image)

        st.subheader("Original Image")

        st.image(
            image_np,
            use_container_width=True
        )

        img = cv2.resize(
            image_np,
            (256, 256)
        )

        img = img.astype(np.float32) / 255.0

        tensor = torch.tensor(
            img.transpose(2, 0, 1),
            dtype=torch.float32
        ).unsqueeze(0).to(device)

        with torch.no_grad():

            output = model(tensor)

            pred = torch.argmax(
                output,
                dim=1
            ).squeeze().cpu().numpy()

        pred = np.clip(
            pred,
            0,
            len(COLORS) - 1
        )

        mask = COLORS[pred]

        st.subheader("Predicted Mask")

        st.image(
            mask,
            use_container_width=True
        )

        mask_large = cv2.resize(
            mask,
            (
                image_np.shape[1],
                image_np.shape[0]
            ),
            interpolation=cv2.INTER_NEAREST
        )

        overlay = cv2.addWeighted(
            image_np,
            0.6,
            mask_large,
            0.4,
            0
        )

        st.subheader("Overlay")

        st.image(
            overlay,
            use_container_width=True
        )


# =====================================
# WEBCAM MODE
# =====================================

elif mode == "Webcam":

    st.subheader(
        "📷 Live Semantic Segmentation"
    )

    webrtc_streamer(
        key="camvid-webcam",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )