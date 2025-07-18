import streamlit as st

st.set_page_config(
    page_title="Model & Training",
    initial_sidebar_state='expanded',
    layout='wide'
)

st.sidebar.markdown("## :material/folder_open: Model & Training Details")
st.sidebar.markdown("""
:material/subdirectory_arrow_right: Model Architectures

> :material/subdirectory_arrow_right: Pre-Trained Models

> :material/subdirectory_arrow_right: Comprehensive Model Details

                    
:material/subdirectory_arrow_right: Transfer Learning Strategy

> :material/subdirectory_arrow_right: Transfer Learning Approach

> :material/subdirectory_arrow_right: Custom Classification Head Arch

                    
:material/subdirectory_arrow_right: Training Environment
                    
> :material/subdirectory_arrow_right: Hardware & Runtime Environment

> :material/subdirectory_arrow_right: Training Config and Hyper-Parameters

>> :material/subdirectory_arrow_right: General Settings

>> :material/subdirectory_arrow_right: Optimizers/ Regularization

>> :material/subdirectory_arrow_right: Callbacks

>> :material/subdirectory_arrow_right: Augmentation
""")


st.title("Model & Training")

# Custom style for subtle accent
st.markdown("""
    <style>
    [data-testid="stExpander"] > details[open] {
        border: 2px solid #1f77b4;
        border-radius: 6px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Model Architectures", "Transfer Learning Strategy", "Training Environment"])

###################################################
# --------- Model Architectures Tab -------------
###################################################
with tab1:
    st.subheader(":blue[:material/category:] Pre-Trained Models")
    st.write("""
    The following pre-trained Computer Vision (CV) models were utilized for this project.
    Each model brings unique architectural features and design philosophies to image recognition.
    """)

    model_cols = st.columns(2)
    for idx in range(4):
        with model_cols[idx % 2]:
            if idx == 0:
                st.info(":material/speed: **EfficientNetB0**")
            elif idx == 1:
                st.success(":material/check_circle: **ResNet-50**")
            elif idx == 2:
                st.warning(":material/storage: **ResNet-101**")
            else:
                st.error(":material/bug_report: **VGG-16**")

    st.divider()

    st.subheader(":blue[:material/info:] Comprehensive Pre-Trained Model Details")

    # -------- EfficientNetB0 --------
    st.info(":material/speed: **EfficientNetB0**")
    st.write("""
    EfficientNetB0 is a compact convolutional neural network introduced as part of the EfficientNet family,  
    designed around compound scaling principles to achieve high accuracy with fewer parameters and FLOPs.
    
    - **Architecture highlights:** Compound scaling of width, depth, and resolution using a scaling coefficient.  
    - **Strengths:** Excellent accuracy-efficiency tradeoff; ideal for deployment on resource-constrained devices.
    - **Paper:** *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks* (M. Tan & Q.V. Le)
    """)

    # -------- ResNet-50 --------
    st.success(":material/check_circle: **ResNet-50**")
    st.write("""
    ResNet-50 is a 50-layer deep residual network introducing **skip connections** to mitigate vanishing gradients, enabling the training of very deep CNNs.

    - **Architecture highlights:** Bottleneck residual blocks; 50 weight layers; skip connections.
    - **Strengths:** Robust general-purpose architecture; widely adopted baseline for vision tasks.
    - **Paper:** *Deep Residual Learning for Image Recognition* (K. He et al.)
    """)

    # -------- ResNet-101 --------
    st.warning(":material/storage: **ResNet-101**")
    st.write("""
    ResNet-101 extends ResNet-50 to 101 layers, deepening the network to capture more complex patterns while retaining the benefits of residual learning.

    - **Architecture highlights:** 101 layers with stacked residual blocks; deeper variant of ResNet-50.
    - **Strengths:** Higher representational capacity; competitive performance on fine-grained tasks.
    - **Trade-offs:** Increased compute and memory usage compared to ResNet-50.
    - **Paper:** *Deep Residual Learning for Image Recognition* (K. He et al.)
    """)

    # -------- VGG-16 --------
    st.error(":material/bug_report: **VGG-16**")
    st.write("""
    VGG-16 is a classic architecture using a stack of 3x3 convolutions and max-pooling layers,  
    known for its simplicity and uniform design.

    - **Architecture highlights:** 16 weighted layers; repeated 3x3 conv filters; no residuals or modern enhancements.
    - **Strengths:** Intuitive structure; effective as a feature extractor.
    - **Trade-offs:** Large number of parameters; computationally expensive relative to accuracy.
    - **Paper:** *Very Deep Convolutional Networks for Large-Scale Image Recognition* (K. Simonyan & A. Zisserman)
    """)

###################################################
# --------- Transfer Learning Tab -------------
###################################################
with tab2:
    st.subheader(":green[:material/swap_vert:] Transfer Learning Approach")

    st.write("""
    Transfer learning enables leveraging rich features learned by large-scale ImageNet pre-trained models, 
    adapting them for domain-specific tasks like waste image classification.

    **Pipeline used:**
    """)
    st.markdown("""
    - **Step 1:** Load ImageNet-pretrained backbone (`include_top=False`) for frozen feature extraction.  
    - **Step 2:** Apply custom `PreProcessingLayer` ensuring input normalization/scaling consistent with pretrained backbone.  
    - **Step 3:** Attach a lightweight classification head tailored for 9 waste categories.  
    - **Step 4:** Train only the classification head; backbone weights remain frozen.
    """)

    st.divider()

    st.subheader(":green[:material/design_services:] Custom Classification Head Architecture")
    st.markdown("The following classification head was adopted on top of the pre-trained models for the Transfer Learning task:")
    st.markdown(
        """
        <div style='text-align: center'>

        Input → PreProcessingLayer  
        ↓  
        Base Model  
        ↓  
        Global Average Pooling  
        ↓  
        Dense(256) with L2 Regularization  
        ↓  
        Batch Normalization  
        ↓  
        ReLU Activation  
        ↓  
        Dropout  
        ↓  
        Dense(9) with Softmax  

        </div>
        """,
        unsafe_allow_html=True
    )
       
    st.divider()

    st.write("""
    Below is a step-wise breakdown of the layers of the classification head as illustrated above.
    """)

    d_cols = st.columns(2)
    with d_cols[0]:
        # Step 1
        with st.expander(":violet[PreProcessing Layer]", icon=":material/tune:"):
            st.write("""
            A Keras preprocessing layer used to normalize pixel values, resize images, or perform other initial transformations required by the base model. This ensures consistent data format and scale.
            """)

        # Step 2
        with st.expander(":green[Base Model]", icon=":material/model_training:"):
            st.write("""
            The frozen convolutional backbone extracted from a pre-trained network (e.g., EfficientNetB0, ResNet-50) trained on ImageNet.
            This network acts as a high-capacity feature extractor, transforming raw image data into high-level feature maps.
            Freezing these layers preserves the model's pre-learned visual understanding while reducing computational cost and overfitting risk.
            """)

        # Step 3
        with st.expander(":orange[Global Average Pooling]", icon=":material/functions:"):
            st.write("""
            Reduces the spatial feature maps from the base model into a 1D feature vector by averaging across spatial dimensions.
            This simplifies the feature representation while preserving essential information.
            """)

        # Step 4
        with st.expander(":blue[Dense (256) + L2 Regularization]", icon=":material/memory:"):
            st.write("""
            Fully connected layer with 256 neurons.
            L2 regularization penalizes large weight magnitudes, helping prevent overfitting.
            """)

    with d_cols[1]:
        # Step 5
        with st.expander(":violet[Batch Normalization]", icon=":material/align_horizontal_center:"):
            st.write("""
            Normalizes the activations of the previous layer within each batch by bringing them to a standard scale and mean. 
            Stabilizing training, enabling higher learning rates, and acts as a form of regularization to improve model generalisation.
            """)

        # Step 6
        with st.expander(":green[ReLU Activation]", icon=":material/bolt:"):
            st.write("""
            Applies a non-linear ReLU function, introducing the ability to model complex feature interactions.
            """)

        # Step 7
        with st.expander(":red[Dropout]", icon=":material/cancel:"):
            st.write("""
            A regularization technique that randomly drops p% of neurons during training to prevent co-adaptation and enhance generalization capabilities.  
            It is only active during training and disabled during inference.  
            """)

        # Step 8
        with st.expander(":blue[Dense (9) + Softmax]", icon=":material/poll:"):
            st.write("""
            Final FC-layer with 9 neurons, corresponding to the 9 waste categories.
            Softmax activation converts the raw logits into a probability distribution over the classes.
            """)

###################################################
# --------- Training Environment Tab -------------
###################################################
with tab3:
    st.subheader(":orange[:material/settings_suggest:] Hardware & Runtime Environment")
    st.markdown("""
    - **Training Location:**  Macbook Pro, Kaggle Notebooks
    - **Hardware:**  
        - **Apple M4Pro Chip** (12-core CPU, 16-core GPU, 24GB unified memory), with Metal Performance Shaders (MPS) acceleration.  
        - **NVIDIA Tesla T4** GPUs (16GB VRAM, 30GB RAM, 4vCPUs)
    - **Mixed Precision:** Enabled using `mixed_float16` policy for faster training and reduced memory usage.  
    - **Framework Versions:**  
        - Python: `3.11.11`  
        - TensorFlow: `2.18.0`  
        - sklearn: `1.2.2`  
    """)

    st.divider()

    st.subheader(":orange[:material/tune:] Training Config & Hyperparameters")
    hp1, hp2, hp3, hp4 = st.tabs([
        "General Settings", "Optimizer/Regularization", "Callbacks", "Augmentation"
    ])

    with hp1:
        st.markdown("""
        - **Input Image Size:**  
            - Raw Image: 524 x 524  
            - Augmented Image: 224 x 224 x 3  
        - **Batch Size:** 5 samples
        - **Epochs:** 100 (with early stopping based on validation loss)  
        - **Total Images:**  
            - Train-set Images: 6068 samples (original + augmented)  
            - Validation-set Images: 764 samples  
        - **Total Waste Categories:** 9  
        """)
    
    with hp2:
        st.markdown("""
        - **Optimizer:** Adam with learning rate `1e-4`; Exponential Moving Average (EMA) enabled  
        - **Loss Function:** Categorical Cross-Entropy  
        - **Regularization Techniques:**  
            - L2 Regularization (`0.0001`) applied to Dense layer  
            - Dropout (`20%`) applied after ReLU activation  
            - Batch Normalization with momentum (`0.99`)  
        - **Activation Functions**
            - **ReLU** Activation for first FC-Layer  
            - **Softmax** Activation for final FC-Layer  
        """)
    
    with hp3:
        st.write("The following training callbacks were set to monitor and modify/adjust the model training process at specific points during training")
        st.markdown("""
        - **`ReduceLROnPlateau`**: Reduces LR by a _factor_ of **`0.4`** if no improvement for _patience_ of **5 epochs**. Minimum LR is set to `1e-6`.  
        - **`EarlyStopping`**: Halts training early if no improvement by _min_delta_ of **`5e-5`** for _patience_ of **7 epochs** in validation set performance. Evaluation starts after **50** epochs.  
        - **`ModelCheckpoint`**: Saves the model's weights during training, typically to preserve the best-performing version.  
        """)
    
    with hp4:
        st.write("To enhance generalization, the following image augmentation techniques were applied to the dataset before training. This expanded the dataset at hand (original + augmented) for better model training")
        aug_configs = {
            "Random Crop": {
                "icon": ":material/crop:",
                "color": "blue",
                "description": "Crops a random portion of the image to `256x256` pixels."
            },
            "Random Zoom": {
                "icon": ":material/zoom_in:",
                "color": "green",
                "description": "Randomly zooms into or out of the image by `0.1`, filling new pixels with a `constant` value."
            },
            "Random Rotation": {
                "icon": ":material/rotate_right:",
                "color": "violet",
                "description": "Randomly rotates the image by `0.1` radians, filling new pixels with a `constant` value."
            },
            "Random Hor-Flip": {
                "icon": ":material/flip:",
                "color": "orange",
                "description": "Randomly flips the image horizontally."
            },
            "Random Contrast": {
                "icon": ":material/contrast:",
                "color": "red",
                "description": "Randomly adjusts the contrast of the image by `0.1`."
            },
            "Random Translation": {
                "icon": ":material/open_with:",
                "color": "blue",
                "description": "Randomly translates the image horizontally and vertically by `0.1`, filling new pixels with a `constant` value."
            },
            "Resizing": {
                "icon": ":material/aspect_ratio:",
                "color": "green",
                "description": "Resizes the image to the target `IMG_SIZE` (e.g., 224x224, 256x256) at the end of the pipeline."
            }
        }

        w_aug_cols = st.columns(3)
        for idx, aug_name in enumerate(aug_configs.keys()):
            config = aug_configs[aug_name]
            with w_aug_cols[idx % 3]:
                # Use f-string for colored icon and text in expander title
                with st.expander(f"{config['icon']} :{config['color']}[**{aug_name}**]"):
                    st.markdown(config['description'])