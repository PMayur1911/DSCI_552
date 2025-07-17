import streamlit as st
import pandas as pd

from data.store import ModelStore
from utils.radar_plot import radar_chart
from utils.summary_metrics import get_summary_metrics

st.set_page_config(
    page_title="Performance & Results Overview",
    initial_sidebar_state='expanded'
)
st.title("Performance & Results Overview")

models = [
    ModelStore("EfficientNetB0"),
    ModelStore("ResNet50"),
    ModelStore("ResNet101"),
    ModelStore("VGG16"),
]

tabs = st.tabs([x.model_name for x in models] + ["Overall Comparison"])

# Custom style for subtle accent
st.markdown("""
    <style>
    [data-testid="stExpander"] > details[open] {
        border: 2px solid #1f77b4;
        border-radius: 6px;
        padding: 10px;
    }
    .st-emphasis {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 0.5rem;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
    hr.custom-hr {
        border: none;
        border-top: 1px solid rgba(255, 255, 255, 0.2);
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

for idx, model in enumerate(models):
    with tabs[idx]:
        st.subheader(f"{model.model_arch.get_icon()} {model.model_name}")

        with st.expander(":green[:material/engineering:] Model Architecture Details"):
            st.markdown(f" **:{model.model_arch.color}[{model.model_name}'s]** architecture retrofitted with the custom classification head:")

            header_cols = st.columns([2, 1, 1])
            header_cols[0].markdown("**Layer (type)**")
            header_cols[1].markdown("**Output Shape**")
            header_cols[2].markdown("**Param #**")

            for (name, layer_type, shape, param) in model.model_arch.arch:
                i_cols = st.columns(1, border=True)
                with i_cols[0]:
                    cols = st.columns([2, 1, 1])
                    cols[0].markdown(f"{name} :blue[({layer_type})]")
                    cols[1].markdown(f":green[{shape}]")
                    cols[2].markdown(f":orange[{param}]")

            st.divider()
            st.markdown(f"**Total Params:** :orange[{model.model_arch.get_params(0)}]")
            st.markdown(f"**Trainable Params:** :orange[{model.model_arch.get_params(1)}]")
            st.markdown(f"**Non-trainable Params:** :orange[{model.model_arch.get_params(2)}]")

        with st.expander(":violet[:material/hourglass_bottom:] Training Summary"):
            st.success(f":material/search_activity: Training stopped at **Epoch {model.model_arch.epoch_stop}** due to EarlyStopping")
            st.image(model.model_arch.loss_img_path, caption=f"Training & Validation Loss Curves for {model.model_name}")

        st.divider()

        # Overall Performance Metrics
        st.markdown("### :green[:material/insights:] Overall Performance Metrics")
        metrics_cols = st.columns(5)
        for idx, (metric, value) in enumerate(model.get_overall_metrics()):
            metrics_cols[idx].metric(f":blue[:material/nearby:] {metric}", value)

        st.markdown("")

        st.subheader(":orange[:material/analytics:] Detailed Performance Metrics")
        perf_tabs = st.tabs(["Training Set", "Validation Set", "Test Set"])

        for i, split in enumerate(["Train", "Val", "Test"]):
            metric_stat = model.train_metrics if split == "Train" else model.val_metrics if split == "Val" else model.test_metrics

            with perf_tabs[i]:
                cols = st.columns([1,1,1,1,1])
                cols[1].metric(":green[:material/target:] Accuracy", metric_stat.get_accuracy())
                cols[3].metric(":orange[:material/inventory_2:] Support", metric_stat.get_support())

                st.markdown("")
                st.write("**Macro-Metrics** (equal class weighting):")
                o_col = st.columns(1, border=True)
                with o_col[0]:
                    macro_cols = st.columns(4)
                    macro_cols[0].metric(":blue[:material/center_focus_strong:] Precision", metric_stat.macro_metrics.get_metrics("precision"))
                    macro_cols[1].metric(":orange[:material/refresh:] Recall", metric_stat.macro_metrics.get_metrics("recall"))
                    macro_cols[2].metric(":violet[:material/balance:] F1-Score", metric_stat.macro_metrics.get_metrics("f1_score"))
                    macro_cols[3].metric(":green[:material/stacked_line_chart:] AUC", metric_stat.macro_metrics.get_metrics("auc"))
                st.markdown("")

                st.write("**Weighted-Metrics** (weighted by support):")
                o_col = st.columns(1, border=True)
                with o_col[0]:
                    weighted_cols = st.columns(4)
                    weighted_cols[0].metric(":blue[:material/center_focus_strong:] Precision", metric_stat.weighted_metrics.get_metrics("precision"))
                    weighted_cols[1].metric(":orange[:material/refresh:] Recall", metric_stat.weighted_metrics.get_metrics("recall"))
                    weighted_cols[2].metric(":violet[:material/balance:] F1-Score", metric_stat.weighted_metrics.get_metrics("f1_score"))
                    weighted_cols[3].metric(":green[:material/stacked_line_chart:] AUC", metric_stat.weighted_metrics.get_metrics("auc"))

                st.divider()

                st.markdown("##### :violet[:material/table:] **Classification Report:**")
                st.dataframe(
                    pd.DataFrame(metric_stat.get_class_report()).style
                    .background_gradient(subset=["Precision", "Recall", "F1-Score"], cmap="RdYlGn")
                    .format({"Precision": "{:.2%}", "Recall": "{:.2%}", "F1-Score": "{:.2%}"}),
                    use_container_width=True
                )
                st.divider()

                auc_scores = metric_stat.get_auc_scores()
                auc_cols = st.columns([2, 3], vertical_alignment="top")
                key = f"{model.model_name}_{split}"


                with auc_cols[0]:
                    st.markdown("##### :green[:material/stacked_line_chart:] **AUC Class-wise Scores:**")
                    st.write(auc_scores)

                with auc_cols[1]:
                    radar_chart(
                        values=auc_scores,
                        classes=list(auc_scores.keys()),
                        key=f"radar_{key}",
                        lr=metric_stat.log_scale
                    )

with tabs[4]:
    st.subheader(":violet[:material/trophy:] Overall Results")
    
    # Training Metrics Summary Table
    st.markdown("##### :green[:material/play_circle:] **Training Metrics Summary**")
    train_df = get_summary_metrics(models, "train")
    st.dataframe(train_df.style.background_gradient(cmap="YlGn"), use_container_width=True)

    # Validation Metrics Summary Table
    st.markdown("##### :orange[:material/tune:] **Validation Metrics Summary**")
    val_df = get_summary_metrics(models, "val")
    st.dataframe(val_df.style.background_gradient(cmap="YlGn"), use_container_width=True)

    # Testing Metrics Summary Table
    st.markdown("##### :blue[:material/fact_check:] **Testing Metrics Summary**")
    test_df = get_summary_metrics(models, "test")
    st.dataframe(test_df.style.background_gradient(cmap="YlGn"), use_container_width=True)

    st.divider()

    st.markdown("### :violet[:material/mystery:] **Key Observations**")

    st.markdown(f"""
    ##### :green[:material/play_circle:] **Training Set Metrics**
    - :{models[2].model_arch.color}[**ResNet101**] achieves near-perfect training performance (Acc ≈ 0.9967, F1 ≈ 0.9967).
    - :{models[1].model_arch.color}[**ResNet50**] is close behind (Acc=0.9934, F1=0.993).
    - :{models[3].model_arch.color}[**VGG16**] performs moderately (F1=0.9565).
    - :{models[0].model_arch.color}[**EfficientNetB0**] shows weakest training performance (F1=0.9031).

    ##### :orange[:material/tune:] **Validation Set Metrics**
    - :{models[2].model_arch.color}[**ResNet101**] continues to lead on validation accuracy and F1-score (Acc=0.9018, F1=0.9014).
    - :{models[1].model_arch.color}[**ResNet50**] very close with nearly identical performance.
    - :{models[3].model_arch.color}[**VGG16**] performs moderately (Acc=0.8494, F1=0.8493).
    - :{models[0].model_arch.color}[**EfficientNetB0**] lags behind (Acc=0.8062, F1=0.8040).

    ##### :blue[:material/fact_check:] **Testing Set Metrics**
    - :{models[1].model_arch.color}[**ResNet50**] outperforms all models on test set (Acc=0.9067, F1=0.9065).
    - :{models[2].model_arch.color}[**ResNet101**] follows closely (Acc=0.8910, F1=0.8914).
    - :{models[3].model_arch.color}[**VGG16**] consistent with validation performance (Acc=0.8333, F1=0.8334).
    - :{models[0].model_arch.color}[**EfficientNetB0**] weakest on test set (Acc=0.7904, F1=0.7873).

    ### 🏆 **Conclusion:**
    ##### :{models[1].model_arch.color}[**ResNet50**] emerges as the best all-rounder with top test performance, strong validation metrics, and competitive training scores.
    """)