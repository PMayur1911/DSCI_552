import streamlit as st
import plotly.graph_objects as go
import numpy as np

def radar_chart(values, classes, key, lr):
        plot_vals = [np.log10(values[c] * 100) for c in classes] + [np.log10(values[classes[0]] * 100)]
        
        fig = go.Figure(
                go.Scatterpolar(
                    r=plot_vals, 
                    theta=classes + [classes[0]], 
                    fill='toself'
            ))

        fig.update_layout(
                polar=dict(
                        radialaxis=dict(
                                visible=False, 
                                range=[lr, 2]
                        )), 
                width=475, 
                height=475, 
                showlegend=False
            )

        st.plotly_chart(fig, use_container_width=True, key=key)
        st.markdown(f'_Radar Scale (in log scale): [{lr}-2.00]_')


def draw_donut(title, values, total, key):
    # Class labels
    labels = [
        "Cardboard", "Food Organics", "Glass", "Metal", "Misc Trash",
        "Paper", "Plastic", "Textile", "Vegetation"
    ]

    streamlit_colors = [
        "#1f77b4",  # Blue
        "#2ca02c",  # Green
        "#9467bd",  # Violet
        "#17becf",  # Teal
        "#bcbd22",  # Olive
        "#ff7f0e",  # Orange
        "#d62728",  # Red
        "#7f7f7f",  # Gray
        "#8c564b"   # Brown
    ]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        textinfo='label+percent',
        marker=dict(colors=streamlit_colors)
    )])

    fig.update_layout(
        title=title,
        annotations=[dict(text=f'{total}<br>samples', x=0.5, y=0.5, font_size=14, showarrow=False)],
        showlegend=False,
        # margin=dict(t=0, b=0, l=0, r=0)
    )

    st.plotly_chart(fig)