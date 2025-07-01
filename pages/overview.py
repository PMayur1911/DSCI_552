import streamlit as st
import pandas as pd
import altair as alt

st.title("Automated Waste Classification using Transfer Learning and Data Augmentation")
st.write("Welcome to the project showcase. Use the sidebar to navigate.")

# --------- Project Objectives -------------
st.header("Project Objectives")
st.write(
    """
    This project demonstrates the application of deep learning and transfer learning to automate waste classification. 
    The goal is to accurately identify waste types from images using pre-trained models, advanced data augmentation, and model optimization techniques. 
    The project also compares the performance of multiple architectures to determine the most suitable model for real-world deployment.
    """
)
st.divider()

waste_classes = {
    "classes": ["Cardboard","Food Organics","Glass","Metal","Misc. Trash","Paper","Plastic","Textile Trash","Vegetation"],
    "counts": [461, 411, 420, 790, 495, 500, 921, 318, 436],
    "icons": [":material/package:", ":material/restaurant:", ":material/wine_bar:", ":material/build:", ":material/delete_outline:", ":material/description:", ":material/recycling:", ":material/checkroom:", ":material/local_florist:"],
    "colors": ["orange", "green", "blue", "blue", "red", "orange", "grey", "violet", "green"]
}

w_classes_df = pd.DataFrame(waste_classes)


# --------- Waste Classes -------------
st.header("Waste Classes")
st.write("The Waste Classification is done across the following 9 categories, which includes organic, biodegradable and recycable groups. These are - ")
# st.write(waste_classes["classes"])

w_cols = st.columns(3, gap='large', vertical_alignment='center')
for idx, row in w_classes_df.iterrows():
    with w_cols[idx % 3]:
        w_class, icon, color = row["classes"], row["icons"], row["colors"]
        st.badge(f"{icon} {w_class}", color=color, width='stretch')

st.divider()

# --------- Dataset Summary -------------
st.header("Dataset Summary")
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

st.write("TBD - Write about Train/Val/Test Splits")

st.divider()


# --------- Dataset Augmentation -------------
st.header("Dataset Image Augmentation")
img_aug = ["Image Crop", "Image Zoom", "Image Rotation", "Image Flip", "Image Constrast", "Image Translation"]
st.write(
    """
    A common practise in Image Classification tasks is to augment the images in the dataset to generate similar looking images and expand the dataset to help the model improve its generalisation capabilities.
    Here, the following image augmentation techniques were applied - 
    """
)
# st.write(img_aug)

# st.divider()

w_aug_cols = st.columns(3, gap='large', vertical_alignment='center')
for idx, aug in enumerate(img_aug):
    with w_aug_cols[idx % 3]:
        with st.expander(f"{aug}"):
            st.write("Lorem Ipsum about the Image Augmentation Technique")
        # w_class, icon, color = row["classes"], row["icons"], row["colors"]
        # st.badge(f"{icon} {w_class}", color=color, width='stretch')