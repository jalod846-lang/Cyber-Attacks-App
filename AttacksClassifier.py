import streamlit as st
import pandas as pd
import numpy as np
import pickle
import random
from datetime import datetime
import io
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Page config
st.set_page_config(
    page_title="Cyber Attacks Classifier",
    layout="wide"
)

# Initialize session state - SIMPLIFIED
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_name' not in st.session_state:
    st.session_state.model_name = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'has_predictions' not in st.session_state:
    st.session_state.has_predictions = False

# Header
st.title(" Cyber Attacks Classifier ")
st.markdown("---")

# Create 2 columns for better layout
col_left, col_right = st.columns(2)

# ==================== LEFT COLUMN - DATA & MODEL ====================
with col_left:
    st.header("📁 Step 1: Upload Data")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose CSV file", type="csv", key="csv_upload")
    
    if uploaded_file is not None:
        try:
            st.session_state.data = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded: {st.session_state.data.shape[0]} rows, {st.session_state.data.shape[1]} columns")
            st.dataframe(st.session_state.data.head(5), use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")
    
    st.markdown("---")
    st.header("🤖 Step 2: Load Model")
    
    # Model upload
    model_file = st.file_uploader("Choose model file", type=['pkl', 'joblib'], key="model_upload")
    
    if model_file is not None:
        try:
            # Try to load the model
            file_content = model_file.getvalue()
            try:
                st.session_state.model = pickle.loads(file_content)
                st.session_state.model_name = "Model (pickle)"
            except:
                try:
                    st.session_state.model = joblib.load(io.BytesIO(file_content))
                    st.session_state.model_name = "Model (joblib)"
                except:
                    st.error("Could not load model")
            
            if st.session_state.model:
                st.success(f"✅ Model loaded: {type(st.session_state.model).__name__}")
        except Exception as e:
            st.error(f"Error: {e}")

# ==================== RIGHT COLUMN - PREDICT & ANALYZE ====================
with col_right:
    st.header("🎲 Step 3: Make Predictions")
    
    # Check if we have data and model
    if st.session_state.data is None:
        st.warning("Please upload data first")
    elif st.session_state.model is None:
        st.warning("Please load a model first")
    else:
        # Check for label column
        if 'Label' in st.session_state.data.columns:
            features = st.session_state.data.drop(columns=['Label'])
            labels = st.session_state.data['Label']
            st.info("✓ Label column found - metrics will be available")
        elif 'target' in st.session_state.data.columns:
            features = st.session_state.data.drop(columns=['target'])
            labels = st.session_state.data['target']
            st.info("✓ Target column found - metrics will be available")
        else:
            features = st.session_state.data
            labels = None
            st.info("ℹ️ No label column - predictions only")
        
        # Number of samples
        max_samples = min(500, len(features))
        num_samples = st.number_input("Number of samples to predict:", 
                                     min_value=1, max_value=len(features), 
                                     value=min(100, max_samples), step=10)
        
        # Predict button
        if st.button("🔮 Make Predictions", type="primary", use_container_width=True):
            try:
                # Select random samples
                indices = random.sample(range(len(features)), num_samples)
                sample_features = features.iloc[indices].copy()
                
                # Make predictions
                with st.spinner("Making predictions..."):
                    predictions = st.session_state.model.predict(sample_features)
                
                # Create results dataframe
                results = sample_features.copy()
                results['Index'] = indices
                results['Prediction'] = predictions
                
                # Add actual labels if available
                if labels is not None:
                    results['Actual'] = labels.iloc[indices].values
                
                results['Time'] = datetime.now().strftime("%H:%M:%S")
                
                # Store in session state
                st.session_state.results_df = results
                st.session_state.has_predictions = True
                
                st.success(f"✅ Predicted {num_samples} samples! Scroll down to see results.")
                
            except Exception as e:
                st.error(f"Prediction error: {e}")
    
    # ========== STEP 4: ANALYZE (ALWAYS VISIBLE) ==========
    st.markdown("---")
    st.header("📊 Step 4: Analyze Results")
    
    # Check if we have predictions
    if st.session_state.has_predictions and st.session_state.results_df is not None:
        results = st.session_state.results_df
        
        st.success(f"✅ Showing predictions for {len(results)} samples")
        
        # Show the results
        with st.expander("👁️ View Predictions", expanded=True):
            st.dataframe(results, use_container_width=True)
        
        # Check if we have actual labels
        if 'Actual' in results.columns:
            st.markdown("### 📈 Performance Metrics")
            
            y_true = results['Actual']
            y_pred = results['Prediction']
            
            # Basic metrics
            accuracy = accuracy_score(y_true, y_pred)
            correct = (y_true == y_pred).sum()
            incorrect = len(y_true) - correct
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{accuracy:.4f}")
            with col2:
                st.metric("Correct", f"{correct}")
            with col3:
                st.metric("Incorrect", f"{incorrect}")
            with col4:
                st.metric("Total", f"{len(y_true)}")
            
            # Advanced metrics
            try:
                precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                
                col_p1, col_p2, col_p3 = st.columns(3)
                with col_p1:
                    st.metric("Precision", f"{precision:.4f}")
                with col_p2:
                    st.metric("Recall", f"{recall:.4f}")
                with col_p3:
                    st.metric("F1-Score", f"{f1:.4f}")
                
                # Confusion Matrix
                with st.expander("🔢 Confusion Matrix"):
                    cm = confusion_matrix(y_true, y_pred)
                    classes = sorted(y_true.unique())
                    cm_df = pd.DataFrame(cm, 
                                       index=[f"Actual: {c}" for c in classes],
                                       columns=[f"Pred: {c}" for c in classes])
                    st.dataframe(cm_df, use_container_width=True)
                
                # Classification Report
                with st.expander("📋 Classification Report"):
                    report = classification_report(y_true, y_pred, zero_division=0)
                    st.text(report)
                    
            except Exception as e:
                st.error(f"Could not calculate metrics: {e}")
        
        else:
            st.info("ℹ️ No actual labels available - showing predictions only")
        
        # Distribution chart
        st.markdown("### 📊 Prediction Distribution")
        dist = results['Prediction'].value_counts().reset_index()
        dist.columns = ['Prediction', 'Count']
        st.bar_chart(dist.set_index('Prediction'))
        st.dataframe(dist, use_container_width=True)
        
        # Download button
        csv = results.to_csv(index=False)
        st.download_button("📥 Download Results", csv, "predictions.csv", "text/csv", use_container_width=True)
        
        # Clear button
        if st.button("🗑️ Clear Results", use_container_width=True):
            st.session_state.has_predictions = False
            st.session_state.results_df = None
            st.rerun()
    
    else:
        st.info("👈 Make predictions first to see results here")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center;'> Cyber Attacks Classifier </div>", unsafe_allow_html=True)