
import streamlit as st
import torch
import numpy as np
from PIL import Image
from unet import UNet

COLORS = np.array([
[128, 64,128],
[244, 35,232],
[70, 70, 70],
[102,102,156],
[190,153,153],
[153,153,153],
[250,170, 30],
[220,220, 0],
[107,142, 35],
[152,251,152],
[70,130,180],
[220, 20, 60],
[255, 0, 0],
[0, 0,142],
[0, 0, 70]
], dtype=np.uint8)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Load Model
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

st.title("U-Net Semantic Segmentation")

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Original Image",
        use_container_width=True
    )

    img = image.resize((256,256))

    img = np.array(img).astype(np.float32)/255.0

    tensor = torch.tensor(
        img.transpose(2,0,1),
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(tensor)

        pred = torch.argmax(
            output,
            dim=1
        )

        mask = pred.squeeze().cpu().numpy()
    colored_mask = COLORS[mask]
    st.image(
        colored_mask,
        caption="Predicted Mask",
        use_container_width=True
    )
