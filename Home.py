import streamlit as st

st.set_page_config(
    page_title="Home - Waste Classification Project",
    initial_sidebar_state='expanded',
    layout='wide'
)

# --- Sidebar TOC ---
st.sidebar.markdown("## :material/folder_open: Home")
st.sidebar.markdown("""
:material/subdirectory_arrow_right: What’s this project about?

:material/subdirectory_arrow_right: What powered this project?

:material/subdirectory_arrow_right: Timeline?

:material/subdirectory_arrow_right: How to navigate this app?
""")

# --- Title ---
st.title(":blue[:material/home:] Transfer Learning for Multi-Class Waste Classification")
st.markdown('Implemented by '
    '[Mayur Prasanna](https://www.linkedin.com/in/pmayur19/) - '
    'view project source code on '
    '[GitHub](http://www.blankwebsite.com/)')

st.divider()

# --- Project Overview ---
cols = st.columns([1])
with cols[0]:
    st.subheader(":green[:material/lightbulb:] What's this project about?")
    st.markdown("""
    Welcome!
    This project explores the use of **deep learning and transfer learning** to classify images of waste into 
    9 categories, encompassing _organic, recyclable and general trash_ - helping move towards smarter and more 
    automated waste management solutions.
                
    I worked on this as part of my coursework at USC - building, training, evaluating and benchmarking several
    pre-trained _Convolutional Neural Network (CNN)_ models to compare their performance on this task.""")

st.divider()

# --- Tools & Technologies Used ---
cols = st.columns([3,2])
with cols[0]:
    st.subheader(":violet[:material/terminal:] What powered this project?")
    st.markdown("""
    - **Libraries:** Python 3.11.11, TensorFlow 2.18, scikit-learn 1.2.2, Streamlit 1.46.1 
    - **Hardware Platforms:**  
        - Apple MacBook Pro M4Pro with Metal Performance Shaders 
        - Kaggle Cloud Notebooks with NVIDIA Tesla T4 GPU  
    - **Techniques:**  
        - Convolutional Neural Networks & Transfer Learning  
        - On-the-fly Image Augmentation  
        - Hyper-parameter Tuning, Training Callbacks & Mixed Precision Training
        - Macro and Weighted Evaluations
    """)

# --- Timeline & Context ---
# cols = st.columns([2,1])
with cols[1]:
    st.subheader(":orange[:material/history:] Timeline?")
    st.markdown("""
    - **University:** University of Southern California (USC Viterbi)  
    - **Course:** DSCI 552 - Machine Learning for Data Science
    - **Professor:** [Prof. Mohammad Reza Rajati](https://viterbi.usc.edu/directory/faculty/Rajati/Mohammad-Reza)
    - **Semester:** Spring 2025  
    - **Project Duration:** 3 weeks  
    """)

st.divider()


# --- Navigation Guide ---
cols = st.columns([1])
with cols[0]:
    st.subheader(":blue[:material/menu_book:] How to navigate this app?")
    st.markdown("""
    This dashboard has a few dedicated pages you can explore:

    - **Overview:** Dataset, objectives, and how I approached the problem.
    - **Model & Training Details:** Pre-Trained architectures, transfer learning strategy, and training environment and setup.
    - **Performance & Results Overview:** Evaluation Metrics per model per train/val/test set, visualizations, and comparison across the board.
    
    ###### :material/double_arrow: Just use the **sidebar on the left to jump between pages!**
    """)

st.divider()

# --- Call to action ---
cols = st.columns([1,10,1], border=False)
with cols[1]:
    st.markdown(
        """
        ##### ✨ Thanks for checking this out — feel free to explore the details, and reach out if you’d like to discuss this project or my work!
        """
    )
