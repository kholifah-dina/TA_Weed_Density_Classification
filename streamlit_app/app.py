import streamlit as st
import pandas as pd
import plotly.express as px
import os

from predict import train_models, test_inference

# Page configuration
st.set_page_config(
    page_title="Weed Density Classification",
    layout="wide",
    page_icon="🌿"
)

# ================= UI STYLE =================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}
            
/* Hide uploader helper text (Limit 200MB • JPG, JPEG) */
[data-testid="stFileUploader"] div div span {
    display: none;
}

:root {
    --primary-green: #2ecc71;
    --dark-green: #27ae60;
    --light-green: #e8f8f5;
}

/* Buttons */
.stButton>button {
    background-color: var(--primary-green);
    color: white;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-weight: 600;
    border: none;
}

.stButton>button:hover {
    background-color: var(--dark-green);
}

/* Gradient Boosting Highlight */
.highlight-gb {
    background-color: var(--light-green);
    border-left: 5px solid var(--primary-green);
    padding: 15px;
    border-radius: 6px;
    font-size: 22px;
    font-weight: bold;
    color: var(--dark-green);
    text-align: center;
    margin: 20px 0;
}

</style>
""", unsafe_allow_html=True)

# ================= SESSION STATE =================
if 'page' not in st.session_state:
    st.session_state['page'] = 'Dashboard'

def change_page(page):
    st.session_state['page'] = page


# ================= DASHBOARD =================
if st.session_state['page'] == 'Dashboard':

    st.markdown("<div style='text-align:center; margin-top:60px;'>", unsafe_allow_html=True)

    st.title("🌿 Weed Density Classification System")
    st.markdown("### Machine Learning-Based Prototype")

    st.write("")

    st.markdown("""
    <div style='font-size:18px; max-width:750px; margin:auto; line-height:1.6; color:#555'>
    
    This prototype system is designed to classify weed density levels in agricultural images 
    using machine learning techniques. 
    
    The system evaluates several classification algorithms including Logistic Regression, 
    Support Vector Machine, Decision Tree, Random Forest, and <b>Gradient Boosting</b>. 
    
    In this research, <b>Gradient Boosting</b> is highlighted as the primary algorithm for 
    performance evaluation.

    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    col1, col2, col3 = st.columns([1,1,1])

    with col2:
        if st.button("🚀 Start System", use_container_width=True):
            change_page("Training")
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ================= MAIN APP =================
else:

    st.markdown("## 🌿 Weed Density Classification System")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📊 Training Page", use_container_width=True):
            change_page("Training")

    with col2:
        if st.button("🎯 Testing Page", use_container_width=True):
            change_page("Testing")

    st.write("---")

    # ================= TRAINING PAGE =================
    if st.session_state['page'] == 'Training':

        st.header("📊 Model Training")

        st.markdown("""
        Upload labeled weed images based on their density class to train the machine learning models.

        **Upload Requirements**
        - File format: JPG only
        - Maximum: 190 images per class
        """)

        max_files = 190

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Renggang")
            renggang_files = st.file_uploader(
                "Upload Images",
                type=['jpg'],
                accept_multiple_files=True,
                key="renggang"
            )

        with col2:
            st.subheader("Sedang")
            sedang_files = st.file_uploader(
                "Upload Images",
                type=['jpg'],
                accept_multiple_files=True,
                key="sedang"
            )

        with col3:
            st.subheader("Padat")
            padat_files = st.file_uploader(
                "Upload Images",
                type=['jpg'],
                accept_multiple_files=True,
                key="padat"
            )

        # Limit upload
        if renggang_files and len(renggang_files) > max_files:
            st.error("Maximum 190 images allowed for Renggang class.")
            renggang_files = renggang_files[:max_files]

        if sedang_files and len(sedang_files) > max_files:
            st.error("Maximum 190 images allowed for Sedang class.")
            sedang_files = sedang_files[:max_files]

        if padat_files and len(padat_files) > max_files:
            st.error("Maximum 190 images allowed for Padat class.")
            padat_files = padat_files[:max_files]

        # Dataset summary
        st.write("### Dataset Summary")

        c_r = len(renggang_files) if renggang_files else 0
        c_s = len(sedang_files) if sedang_files else 0
        c_p = len(padat_files) if padat_files else 0
        total = c_r + c_s + c_p

        df_counts = pd.DataFrame({
            "Class":["Renggang","Sedang","Padat","Total"],
            "Images":[c_r, c_s, c_p, total]
        })

        st.table(df_counts.set_index("Class"))

        # Training button
        all_uploaded = c_r > 0 and c_s > 0 and c_p > 0

        if all_uploaded:

            if st.button("Train Model", use_container_width=True):

                with st.spinner("Running preprocessing, feature extraction, and model training..."):

                    dataset_dict = {
                        "Renggang":[f.read() for f in renggang_files],
                        "Sedang":[f.read() for f in sedang_files],
                        "Padat":[f.read() for f in padat_files]
                    }

                    saved_info = train_models(dataset_dict)

                    st.success("Model training completed successfully. The trained models have been saved.")

                    st.write("### Selected Features")

                    sf_df = pd.DataFrame({
                        "Feature": saved_info['features'],
                        "Selected": ["Yes"]*len(saved_info['features'])
                    })

                    st.table(sf_df.set_index("Feature"))

        else:
            st.warning("Please upload at least one image for each class before training.")


    # ================= TESTING PAGE =================
    elif st.session_state['page'] == 'Testing':

        st.header("🎯 Model Testing")

        st.markdown("""
        Upload a weed image to classify its density level using the trained machine learning models.
        """)

        model_path = "models/weed_models.joblib"

        if not os.path.exists(model_path):

            st.error("No trained models found. Please train the models first.")

        else:

            if 'model_loaded' not in st.session_state:
                st.session_state['model_loaded'] = False

            if st.button("Load Model"):
                st.session_state['model_loaded'] = True
                st.success("Latest trained models loaded successfully.")

            if st.session_state['model_loaded']:

                st.write("### Upload Test Image")

                test_image = st.file_uploader(
                    "Upload an image (JPG only)",
                    type=['jpg'],
                    accept_multiple_files=False
                )

                if test_image:

                    st.image(test_image, width=300, caption="Uploaded Image")

                    if st.button("Run Testing", use_container_width=True):

                        with st.spinner("Running prediction..."):

                            preds, saved_data = test_inference(test_image.read())

                            gb_result = preds['gradient_boosting']

                            st.markdown(
                                f"<div class='highlight-gb'>Prediction Result<br>Predicted Density Level: {gb_result.upper()}<br>Model: Gradient Boosting</div>",
                                unsafe_allow_html=True
                            )

                            metrics_data = saved_data['metrics']

                            model_names = {
                                "logistic_regression":"Logistic Regression",
                                "svm":"Support Vector Machine",
                                "decision_tree":"Decision Tree",
                                "random_forest":"Random Forest",
                                "gradient_boosting":"Gradient Boosting"
                            }

                            rows = []

                            for key,val in metrics_data.items():

                                rows.append({
                                    "Model":model_names[key],
                                    "Accuracy":val["Accuracy"],
                                    "Precision":val["Precision"],
                                    "Recall":val["Recall"],
                                    "F1-score":val["F1-score"]
                                })

                            df_comp = pd.DataFrame(rows)

                            st.write("### Model Accuracy Comparison")

                            def highlight_gb(row):
                                if row['Model']=="Gradient Boosting":
                                    return ['background-color:#1f3d2b; color:#2ecc71; font-weight:bold']*len(row)
                                return ['']*len(row)

                            st.dataframe(
                                df_comp[['Model','Accuracy']].style.apply(highlight_gb,axis=1).hide(axis="index"),
                                use_container_width=True
                            )

                            # Metrics visualization
                            st.write("### Model Performance Comparison")

                            df_melt = df_comp.melt(id_vars="Model",
                                                   var_name="Metric",
                                                   value_name="Score")

                            fig = px.bar(
                                df_melt,
                                x="Model",
                                y="Score",
                                color="Model",
                                facet_col="Metric",
                                title="Model Evaluation Metrics"
                            )

                            fig.update_layout(showlegend=False)

                            st.plotly_chart(fig, use_container_width=True)

                            # Confusion matrix
                            st.write("### Confusion Matrix - Gradient Boosting")

                            cm = saved_data['confusion_matrices']['gradient_boosting']['matrix']
                            labels = saved_data['confusion_matrices']['gradient_boosting']['labels']

                            fig_cm = px.imshow(
                                cm,
                                text_auto=True,
                                color_continuous_scale="Greens",
                                x=labels,
                                y=labels
                            )

                            fig_cm.update_layout(
                                xaxis_title="Predicted",
                                yaxis_title="Actual"
                            )

                            st.plotly_chart(fig_cm, use_container_width=True)