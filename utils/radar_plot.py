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