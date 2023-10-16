import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Load the data
df = pd.read_csv("FinalCSV_MTS.csv")

def plot_surface_from_model(model, df, title):
    u = np.linspace(df['M_TeamConfidence_VerticalUp'].min(), df['M_TeamConfidence_VerticalUp'].max(), 100)
    v = np.linspace(df['M_TeamConfidence_VerticalDown'].min(), df['M_TeamConfidence_VerticalDown'].max(), 100)
    U, V = np.meshgrid(u, v)

    df_temp = pd.DataFrame({
        'M_TeamConfidence_VerticalUp': U.ravel(),
        'M_TeamConfidence_VerticalDown': V.ravel(),
        'M_TeamConfidence_VerticalUp_Squared': U.ravel()**2,
        'M_TeamConfidence_VerticalDown_Squared': V.ravel()**2
    })
    df_temp['M_TeamConfidence_VerticalUp:M_TeamConfidence_VerticalDown'] = df_temp['M_TeamConfidence_VerticalUp'] * df_temp['M_TeamConfidence_VerticalDown']

    predicted_values = model.predict(df_temp)
    Predicted_Surface = predicted_values.values.reshape(U.shape)

    fig = go.Figure(data=[go.Surface(z=Predicted_Surface, x=U, y=V, colorscale='Viridis')])
    fig.update_layout(title=title, scene=dict(
        xaxis_title='M_TeamConfidence_VerticalUp',
        yaxis_title='M_TeamConfidence_VerticalDown'
    ))
    
    return fig

# For Performance:
formulaPos_Performance = "M_PerformancePos ~ M_TeamConfidence_VerticalUp + M_TeamConfidence_VerticalDown + M_TeamConfidence_VerticalUp_Squared + M_TeamConfidence_VerticalDown_Squared + M_TeamConfidence_VerticalUp:M_TeamConfidence_VerticalDown"
formulaNeg_Performance = "M_PerformanceNeg ~ M_TeamConfidence_VerticalUp + M_TeamConfidence_VerticalDown + M_TeamConfidence_VerticalUp_Squared + M_TeamConfidence_VerticalDown_Squared + M_TeamConfidence_VerticalUp:M_TeamConfidence_VerticalDown"
formulaTot_Performance = "M_PerformanceTot ~ M_TeamConfidence_VerticalUp + M_TeamConfidence_VerticalDown + M_TeamConfidence_VerticalUp_Squared + M_TeamConfidence_VerticalDown_Squared + M_TeamConfidence_VerticalUp:M_TeamConfidence_VerticalDown"

# Panel 1
st.title("Performance Models")
modelPos_Performance = smf.ols(formulaPos_Performance, data=df).fit()
modelNeg_Performance = smf.ols(formulaNeg_Performance, data=df).fit()
modelTot_Performance = smf.ols(formulaTot_Performance, data=df).fit()

st.plotly_chart(plot_surface_from_model(modelPos_Performance, df, 'Performance Positive'))
st.plotly_chart(plot_surface_from_model(modelNeg_Performance, df, 'Performance Negative'))
st.plotly_chart(plot_surface_from_model(modelTot_Performance, df, 'Performance Total'))

# For Lead Changes:
formulaPos_Lead = "M_LeadChangesPos ~ M_TeamConfidence_VerticalUp + M_TeamConfidence_VerticalDown + M_TeamConfidence_VerticalUp_Squared + M_TeamConfidence_VerticalDown_Squared + M_TeamConfidence_VerticalUp:M_TeamConfidence_VerticalDown"
formulaNeg_Lead = "M_LeadChangesNeg ~ M_TeamConfidence_VerticalUp + M_TeamConfidence_VerticalDown + M_TeamConfidence_VerticalUp_Squared + M_TeamConfidence_VerticalDown_Squared + M_TeamConfidence_VerticalUp:M_TeamConfidence_VerticalDown"
formulaTot_Lead = "M_LeadChangesTot ~ M_TeamConfidence_VerticalUp + M_TeamConfidence_VerticalDown + M_TeamConfidence_VerticalUp_Squared + M_TeamConfidence_VerticalDown_Squared + M_TeamConfidence_VerticalUp:M_TeamConfidence_VerticalDown"

# Panel 2
st.title("Lead Changes Models")
modelPos_Lead = smf.ols(formulaPos_Lead, data=df).fit()
modelNeg_Lead = smf.ols(formulaNeg_Lead, data=df).fit()
modelTot_Lead = smf.ols(formulaTot_Lead, data=df).fit()

st.plotly_chart(plot_surface_from_model(modelPos_Lead, df, 'Lead Changes Positive'))
st.plotly_chart(plot_surface_from_model(modelNeg_Lead, df, 'Lead Changes Negative'))
st.plotly_chart(plot_surface_from_model(modelTot_Lead, df, 'Lead Changes Total'))