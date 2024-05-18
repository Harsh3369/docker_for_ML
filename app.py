import streamlit as st
import pandas as pd
from PIL import Image
import pickle
import shap
import matplotlib.pyplot as plt

# Load the trained classifier model
with open("physician_claasifier_model_files/output_folder/model/physician_conversion.pkl", "rb") as f:
    conversion_classifer = pickle.load(f)

# Load the validation dataset
df_validation = pd.read_csv('physician_claasifier_model_files/output_folder/Validation_data.csv')
df_validation.drop(['Unnamed: 0'], axis=1, inplace=True)

def inference_output(df_inference, df_validation, conversion_classifier, n):
    df_inference.drop(['Unnamed: 0'], axis=1, inplace=True)

    # Drop ID columns for inference
    drop_id_col_list = ['NPI_ID', 'HCP_ID']
    X_inference = df_inference.drop(drop_id_col_list, axis=1)

    # Make predictions
    df_inference['Prediction'] = conversion_classifier.predict(X_inference)

    # Prepare validation set
    X_validation = df_validation.drop(['NPI_ID', 'HCP_ID', 'TARGET'], axis=1)
    X_validation = X_validation.astype(float)

    # Explain the model using SHAP
    explainer = shap.TreeExplainer(conversion_classifier)
    shap_values = explainer.shap_values(X_inference)

    # Plot SHAP feature importance
    shap.summary_plot(shap_values, X_inference, plot_type='bar')
    plt.savefig('shap_bar_plot.png', bbox_inches='tight')
    fig_1 = Image.open('shap_bar_plot.png')

    shap.summary_plot(shap_values, X_inference)
    plt.savefig('shap_bar_plot_fig_3.png', bbox_inches='tight')
    fig_3 = Image.open('shap_bar_plot_fig_3.png')

    # Extract top features for each prediction
    df = df_inference.loc[df_inference['Prediction'] == 1.0].reset_index(drop=True)
    id_col_list = ['NPI_ID', 'HCP_ID', 'Prediction']
    top_features_df = pd.DataFrame(index=df.index)

    for row_idx in range(len(df)):
        shap_values_row = shap_values[row_idx]
        abs_shap_values = abs(shap_values_row)
        top_feature_indices = abs_shap_values.argsort()[-n:][::-1]
        top_feature_names = df.drop(id_col_list, axis=1).columns[top_feature_indices]

        for col in id_col_list:
            top_features_df.loc[row_idx, col] = df.loc[row_idx, col]

        for i in range(n):
            top_features_df.loc[row_idx, f'REASON{i+1}'] = top_feature_names[i]

    return df_inference, fig_1, fig_3, top_features_df

def main():
    st.markdown("""
    <style>
    body {
        background-color: #000000;
    }
    .side-by-side {
        display: flex;
    }
    .side-by-side > * {
        flex: 1;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    logo_path = "physician_claasifier_model_files/Input_data/propensity-chart.gif"
    logo_image = Image.open(logo_path)
    st.image(logo_image, use_column_width=True)

    st.title("Physician Conversion Application")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        with st.expander("Uploaded CSV Data"):
            st.write(df)

        n = st.selectbox("Select the number of top features", [1, 2, 3, 4, 5])

        df_output, fig_1, fig_3, df_recommendation = inference_output(df, df_validation, conversion_classifer, n)

        st.write("Inference Data with Prediction:")
        st.write(df_output)

        st.write("SHAP PLOTS:")
        st.markdown('<div class="side-by-side">', unsafe_allow_html=True)
        st.markdown('<div>', unsafe_allow_html=True)
        st.write("SHAP BAR PLOT:")
        st.image(fig_1, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div>', unsafe_allow_html=True)
        st.write("SHAP SUMMARY PLOT:")
        st.image(fig_3, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.write("Inference Data with Recommendation:")
        st.write(df_recommendation)

if __name__ == "__main__":
    main()
