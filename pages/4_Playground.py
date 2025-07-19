import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import time

from utils.model_loader import load_trained_model, preprocess_image

st.set_page_config(page_title="Playground", initial_sidebar_state='expanded', layout='wide')
st.title(":green[:material/toys:] Playground — Try the Models!")

# Sidebar TOC
st.sidebar.markdown("## :material/folder_open: Playground")
st.sidebar.markdown("""
:material/subdirectory_arrow_right: Upload Image

:material/subdirectory_arrow_right: Select Models

:material/subdirectory_arrow_right: Results
""")


st.divider()
cols = st.columns([4,4,1], gap='large')
# Image Upload
with cols[0]:
    st.subheader(":blue[:material/upload_file:] Upload an image")
    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
    image = None

    st.markdown("\n\n")

    st.subheader(":violet[:material/graph_2:] Select Models")
    model_options = ["EfficientNetB0", "ResNet50", "ResNet101", "VGG16"]
    selected_models = None
    selected_models = st.multiselect("Select one or more models to run inference", model_options)

# Model selection
with cols[1]:
    # st.subheader(":green[:material/select_all:] Select Models")
    # model_options = ["EfficientNetB0", "ResNet50", "ResNet101", "VGG16"]
    # selected_models = None
    # selected_models = st.multiselect("Select one or more models to run inference", model_options)
    if uploaded_file:
        st.subheader(":orange[:material/select_all:] Preview")
        image = Image.open(uploaded_file)
        img_cols = st.columns([1,4,1])
        img_cols[0].empty()
        with img_cols[1]:
            st.image(image, caption="Uploaded Image Preview", use_container_width=True)
        img_cols[2].empty()

with cols[2]:
    st.empty()

st.divider()

# Inference
if image and selected_models:
    if st.button("**:orange[Run Inference]**", use_container_width=True):

        # Results layout
        st.markdown("### :violet[:material/analytics: Results]")
        res_cols = st.columns(len(selected_models), gap='large')

        for idx, model_name in enumerate(selected_models):
            with res_cols[idx]:
                st.subheader(f":orange[:material/model_training:] {model_name}")
                model = load_trained_model(model_name)
                
                p_image = preprocess_image(image)
                
                start_time = time.time()
                preds = model.predict(p_image)[0]
                end_time = time.time()
                
                inf_time = end_time - start_time
                
                top_idx = np.argmax(preds)
                top_prob = preds[top_idx]
                
                waste_classes = ["Cardboard", "Food Organics", "Glass", "Metal", "Miscellaneous Trash", "Paper", "Plastic", "Textile Trash", "Vegetation"]

                st.markdown(f":green[:material/target:] Predicted Label: `{waste_classes[top_idx]}`")
                st.markdown(f":green[:material/pace:] Inference time: `{inf_time:.2f} sec`")
                st.progress(top_prob.item(), text=f"Confidence: {top_prob*100:.1f}%")

                st.write(":green[:material/bar_chart:] Class probabilities:")
                prob_df = pd.DataFrame({"Class": waste_classes, "Confidence": preds})
                st.bar_chart(prob_df.set_index("Class"))

st.divider()
