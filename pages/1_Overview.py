import streamlit as st
import pandas as pd
import altair as alt

from utils.plotting import draw_donut

st.set_page_config(
    page_title="Overview",
    initial_sidebar_state='expanded',
    layout='centered'
)

st.sidebar.markdown("## :material/folder_open: Overview")
st.sidebar.markdown("""
:material/subdirectory_arrow_right: Objectives

:material/subdirectory_arrow_right: Waste Classes
                    
:material/subdirectory_arrow_right: Dataset Summary

:material/subdirectory_arrow_right: Data Split & Preprocessing
""")


# st.title("Automated Waste Classification using Transfer Learning and Data Augmentation")
# st.title("Transfer Learning for Multi-Class Waste Classification: Project Overview")
st.title("Project Overview")

# --------- Project Objectives -------------
st.subheader(":green[:material/flag:] **Objectives**")
st.markdown("""
- This project aims to build an efficient image classification system to automatically categorize waste into 9 classes using **transfer learning** techniques.  
- Pre-trained models (EfficientNetB0, ResNet50, ResNet101, VGG16) were used as frozen backbones with custom heads fine-tuned on this dataset.  
- Data augmentation and regularization (L2 penalty, batch normalization, dropout) techniques were adopted to improve generalization performance.  
- Results are comprehensively evaluated across training, validation, and test sets.
""")

st.divider()


# --------- Waste Classes -------------
st.subheader(":red[:material/delete_outline:] Waste Classes")
st.write("The Waste Classification is done across the following 9 categories, which includes organic, biodegradable and recycable groups. These are - ")

waste_classes = {
    "classes": ["Cardboard","Food Organics","Glass","Metal","Misc. Trash","Paper","Plastic","Textile Trash","Vegetation"],
    "counts": [461, 411, 420, 790, 495, 500, 921, 318, 436],
    "icons": [":material/package:", ":material/restaurant:", ":material/wine_bar:", ":material/build:", ":material/delete_outline:", ":material/description:", ":material/recycling:", ":material/checkroom:", ":material/local_florist:"],
    "colors": ["orange", "green", "blue", "blue", "red", "orange", "grey", "violet", "green"]
}

w_classes_df = pd.DataFrame(waste_classes)

w_cols = st.columns(3, gap='large', vertical_alignment='center')
for idx, row in w_classes_df.iterrows():
    with w_cols[idx % 3]:
        w_class, icon, color = row["classes"], row["icons"], row["colors"]
        st.button(f":{color}[{icon} **{w_class}**]", use_container_width=True, disabled=True)
        # st.popover(f":{color}[{icon} {w_class}]", use_container_width=True, disabled=True)
        # st.badge(f"{icon} {w_class}", color=color, width='stretch')

st.divider()

# --------- Dataset Summary -------------
st.subheader(":blue[:material/bar_chart:] Dataset Summary")
st.markdown(
    """
    The dataset was specifically curated under the DSCI 552 - Data Science for Machine Learning Course at USC Viterbi (Spring '25).
    

    This includes a total of **_4752_** images across the above 9 waste categories. Each image is a 3-channel RGB and (524 x 524) size.
    Below is the dataset split across each class -
    """
)

bar_chart = alt.Chart(w_classes_df).mark_bar(opacity=0.75).encode(
    x=alt.X('counts', title='Waste Count'),
    y=alt.Y('classes', title='Waste Class', sort=None),
    color=alt.Color('classes', scale=alt.Scale(domain=w_classes_df['classes'], range=w_classes_df['colors']), legend=None),
    tooltip=['classes', 'counts']
).properties(
    width=600,
    height=400
)

st.altair_chart(bar_chart)

st.divider()

# --------- Dataset Splits -------------
st.subheader(":orange[:material/tune:] Data Split & Preprocessing")
st.write("""
- The dataset comprises images from 9 distinct waste categories.
- **Preprocessing applied:**
  - Resized all images to 224x224 pixels for uniformity and compatibility with ImageNet pre-trained backbones.
  - One-hot encoded target labels to represent multi-class categories.
- **Split strategy:**
  - **Training Set:** First 80% of images in each class folder.
  - **Validation Set:** 20% subset randomly selected from training set (stratified per class).
  - **Test Set:** Remaining 20% images from each class folder.
""")


# Distribution counts from image
train_counts = [294, 262, 268, 505, 316, 320, 588, 203, 278]
val_counts   = [74, 66, 68, 127, 80, 80, 148, 51, 70]
test_counts  = [93, 83, 84, 158, 99, 100, 185, 64, 88]

col1, col2, col3 = st.columns(3)

with col1:
    draw_donut("Train Set", train_counts, sum(train_counts), key="train")

with col2:
    draw_donut("Validation Set", val_counts, sum(val_counts), key="val")

with col3:
    draw_donut("Test Set", test_counts, sum(test_counts), key="test")