import streamlit as st

st.title("Model & Training Details")
tab1, tab2, tab3 = st.tabs(["Model Architectures", "Transfer Learning Strategy", "Training Environment"])

###################################################
# --------- Model Architecture -------------
###################################################
with tab1:
    st.subheader("Pre-Trained Models")
    st.write(
        """
        The following pre-trained Computer Vision (CV) models were utilized for the waste classification task.
        Each model contributes unique strengths, enabling a comprehensive performance comparison.
        """
    )

    model_cols = st.columns(2, gap='small', vertical_alignment='center')
    for idx in range(4):
        with model_cols[idx % 2]:
            if idx == 0:
                st.info("###### :material/speed: EfficientNetB0")
            elif idx == 1:
                st.success("###### :material/check_circle: ResNet-50")
            elif idx == 2:
                st.warning("###### :material/storage: ResNet-101")
            else:
                st.error("###### :material/bug_report: VGG-16")
    st.divider()

    st.subheader("Comprehensive Pre-Trained Model Details")

    # --------- EfficientNetB0 ----------
    st.info("###### :material/speed: EfficientNetB0")
    st.write(
        """
        Lightweight and optimized for efficiency, making it ideal for real-time inference on resource-constrained devices.
        """
    )
    with st.expander(":material/settings: View Detailed Specifications for EfficientNetB0"):
        st.markdown(
            """
            **Architecture:**  
            - Compound scaling of depth, width, and resolution.  
            - Efficient feature extraction with fewer parameters than traditional CNNs.  

            **Strengths:**  
            - State-of-the-art accuracy for its size.  
            - Excellent for edge deployment and fast inference.  

            **Trade-offs:**  
            - Slightly lower peak accuracy compared to larger, deeper models.  

            **Paper:** *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*  
            """
        )
        st.write("**Model Parameter Summary:**")
        st.table({
            "Parameter Type": ["Total Parameters", "Trainable Parameters", "Non-Trainable Parameters"],
            "Count": ["4,380,844", "330,761", "4,050,083"]
        })
    st.divider()

    # --------- ResNet-50 ----------
    st.success("###### :material/check_circle: ResNet-50")
    st.write(
        """
        A widely-used deep CNN with residual connections that facilitates training deeper networks while maintaining strong generalization.
        """
    )
    with st.expander(":material/settings: View Detailed Specifications for ResNet-50"):
        st.markdown(
            """
            **Architecture:**  
            - 50-layer deep convolutional network with residual (skip) connections.  

            **Strengths:**  
            - Balanced accuracy and efficiency.  
            - Robust performance for complex image classification tasks.  

            **Trade-offs:**  
            - Moderate memory and compute requirements.  

            **Paper:** *Deep Residual Learning for Image Recognition*, Kaiming He et al.  
            """
        )
        st.write("**Model Parameter Summary:**")
        st.table({
            "Parameter Type": ["Total Parameters", "Trainable Parameters", "Non-Trainable Parameters"],
            "Count": ["24,115,593", "527,369", "23,588,224"]
        })

    st.divider()

    # --------- ResNet-101 ----------
    st.warning("###### :material/storage: ResNet-101")
    st.write(
        """
        A deeper variant of ResNet-50, capable of capturing more complex features, with increased computational demands.
        """
    )
    with st.expander(":material/settings: View Detailed Specifications for ResNet-101"):
        st.markdown(
            """
            **Architecture:**  
            - 101-layer deep CNN with stacked residual blocks.  

            **Strengths:**  
            - Enhanced feature learning for fine-grained classification.  
            - Higher potential accuracy on complex datasets.  

            **Trade-offs:**  
            - Significantly higher parameter count and resource requirements.  

            **Paper:** *Deep Residual Learning for Image Recognition*, Kaiming He et al.  
            """
        )
        st.write("**Model Parameter Summary:**")
        st.table({
            "Parameter Type": ["Total Parameters", "Trainable Parameters", "Non-Trainable Parameters"],
            "Count": ["43,713,929", "1,054,729", "42,659,200"]
        })

    st.divider()

    # --------- VGG-16 ----------
    st.error("###### :material/bug_report: VGG-16")
    st.write(
        """
        A classic CNN known for its simplicity and uniform architecture, still effective for transfer learning despite being parameter-heavy.
        """
    )
    with st.expander(":material/settings: View Detailed Specifications for VGG-16"):
        st.markdown(
            """
            **Architecture:**  
            - 16-layer CNN with small (3x3) convolutional filters stacked in depth.  

            **Strengths:**  
            - Straightforward design ideal for educational purposes and feature extraction.  

            **Trade-offs:**  
            - Large parameter count relative to modern alternatives.  
            - Computationally expensive.  

            **Paper:** *Very Deep Convolutional Networks for Large-Scale Image Recognition*, Simonyan & Zisserman.  
            """
        )
        st.write("**Model Parameter Summary:**")
        st.table({
            "Parameter Type": ["Total Parameters", "Trainable Parameters", "Non-Trainable Parameters"],
            "Count": ["14,849,353", "134,153", "14,715,200"]
        })

    st.divider()


###################################################
# --------- Transfer Learning Strategy-------------
###################################################
with tab2:
    
    st.subheader("Transfer Learning")

    st.write(
        """
        Transfer learning enables leveraging the knowledge from large-scale datasets like ImageNet, allowing efficient adaptation to domain-specific tasks such as waste classification.
        
        In this project, transfer learning was applied systematically to build robust, efficient models:
        """
    )

    st.markdown(
        """
        **Step 1:** Load pre-trained ImageNet models (`include_top=False`) to serve as frozen feature extractors.  
        **Step 2:** Apply a custom `PreProcessingLayer` to ensure model-specific input scaling.  
        **Step 3:** Add a lightweight, regularized custom-classification head tailored for our 9 waste categories.  
        **Step 4:** Fine-tune only the final classification head to adapt the model to the dataset.  
        """
    )
    st.divider()
    
    st.subheader("Custom-Classification Head Architecture")
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
    Below is a step-wise breakdown of the classification head added on top of each pre-trained model.
    Click on each component to view its technical details and purpose.
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


#############################################################
# --------- Hyper-Parameters & Training Details -------------
#############################################################
with tab3:
    st.subheader("Hardware & Runtime Environment")
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

    st.subheader("Training Config and Hyper-Parameters")
    ctab1, ctab2, ctab3, ctab4 = st.tabs(["Training Parameters", "Optimizer and Regularizer", "Callbacks", "Data Augmentation"])

    with ctab1:
        st.markdown("""
        - **Input Image Size:**  
            - Raw Image: 524 x 524  
            - Augmented Image: 224 x 224 x 3  
        - **Batch Size:** 5  
        - **Epochs:** 100 (with early stopping based on validation loss)  
        - **Total Images:**  
            - Train-set Images: 6068 samples (original + augmented)  
            - Validation-set Images: 764 samples  
        - **Total Waste Categories:** 9  
        """)
    
    with ctab2:
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
    
    with ctab3:
        st.write("The following training callbacks were set to monitor and modify/adjust the model training process at specific points during training")
        st.markdown("""
        - **`ReduceLROnPlateau`**: Reduces LR by a _factor_ of **`0.4`** if no improvement for _patience_ of **5 epochs**. Minimum LR is set to `1e-6`.  
        - **`EarlyStopping`**: Halts training early if no improvement by _min_delta_ of **`5e-5`** for _patience_ of **7 epochs** in validation set performance. Evaluation starts after **50** epochs.  
        - **`ModelCheckpoint`**: Saves the model's weights during training, typically to preserve the best-performing version.  
        """)
    
    with ctab4:
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