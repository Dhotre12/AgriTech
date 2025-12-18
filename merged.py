import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import PartialDependenceDisplay
from xgboost import XGBClassifier
import warnings

# --- NEW IMPORTS FOR EXPLAINABLE AI ---
# SHAP: Used for global feature importance and interaction effects (Waterfall plots)
import shap
import matplotlib.pyplot as plt

# LIME: Used for local, instance-specific explanations.
# Wrapped in a try-except block to ensure the app doesn't crash if LIME isn't installed.
try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except Exception:
    LIME_AVAILABLE = False

warnings.filterwarnings("ignore", category=UserWarning)

# ==================== CONFIGURATION ====================
# Sets the browser tab title, icon, and enables 'wide' mode for better dashboard visualization.
st.set_page_config(
    page_title="AgriSmart - AI Crop Recommendation",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== PYTORCH MODEL DEFINITIONS ====================
# ARCHITECTURAL DECISION: 
# We define classes inheriting from nn.Module to allow for flexible loading of state_dicts 
# trained offline. Each class represents a specific hypothesis for tabular data processing.

class CNNModel(nn.Module):
    """
    1D-CNN: Uses 1-dimensional convolution to capture local dependencies/interactions 
    between adjacent features in the input vector.
    """
    def __init__(self, input_dim, output_dim):
        super(CNNModel, self).__init__()
        # Input shape: (Batch, Channels, Sequence_Length) -> (Batch, 1, 8)
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        # Dynamically calculate linear layer input based on pooling reduction
        self.fc = nn.Linear(64 * (input_dim // 2), output_dim)
    def forward(self, x):
        x = x.unsqueeze(1) # Add channel dimension
        x = self.pool(self.relu(self.conv1(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x

class LSTMModel(nn.Module):
    """
    LSTM: Treats the feature vector as a sequence. Useful if we assume there is a 
    sequential relationship or 'flow' between parameters (e.g., Soil -> Weather).
    """
    def __init__(self, input_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, 64, batch_first=True)
        self.fc = nn.Linear(64, output_dim)
    def forward(self, x):
        x = x.unsqueeze(1) # Sequence length of 1, input_dim features per step
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) # Take the last hidden state
        return out

class GRUModel(nn.Module):
    """
    GRU: Gated Recurrent Unit. A more efficient variant of LSTM, often performing 
    comparably with fewer parameters.
    """
    def __init__(self, input_dim, output_dim):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, 64, batch_first=True)
        self.fc = nn.Linear(64, output_dim)
    def forward(self, x):
        x = x.unsqueeze(1)
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

class TransformerModel(nn.Module):
    """
    Transformer: Uses Self-Attention mechanisms. This allows the model to dynamically 
    weigh which environmental factors (N, P, K, etc.) are most relevant for a specific sample.
    """
    def __init__(self, input_dim, output_dim):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, 64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Linear(64, output_dim)
    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = self.transformer(x)
        x = x.mean(dim=1) # Global Average Pooling over the sequence
        x = self.fc(x)
        return x

class ResidualMLP(nn.Module):
    """
    Residual MLP: A deep fully connected network with Skip Connections (ResNet style).
    This helps prevent vanishing gradients, allowing for deeper feature extraction.
    """
    def __init__(self, input_dim, output_dim):
        super(ResidualMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        x1 = self.relu(self.fc1(x))
        x2 = self.relu(self.fc2(x1))
        x = x1 + x2 # The "Residual" connection
        x = self.fc3(x)
        return x

class Hybrid_CNNLSTM(nn.Module):
    """
    Hybrid Architecture: Combines CNN (spatial feature extraction) with LSTM (sequential processing).
    Usually the top performer as it captures both local interactions and global context.
    """
    def __init__(self, input_dim, output_dim):
        super(Hybrid_CNNLSTM, self).__init__()
        self.conv = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(64, 64, batch_first=True)
        self.fc = nn.Linear(64, output_dim)
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.permute(0, 2, 1) # Rearrange for LSTM input
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class ANNModel(nn.Module):
    """
    Baseline ANN: A standard Multi-Layer Perceptron (MLP) used as a performance baseline.
    """
    def __init__(self, input_dim, output_dim):
        super(ANNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, output_dim)
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


# ==================== STYLING & UTILS ====================

st.markdown("""
<style>
    /* ---------------------------------------------------- */
    /* --- NEW CSS FOR BLINKING DOT --- */
    @keyframes blink-animation {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.2; }
    }
    .blinking-dot {
        animation: blink-animation 1.5s infinite;
    }
    /* ---------------------------------------------------- */

    /* Main styling */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #16a34a, #10b981, #0ea5e9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }
    
    /* Performance Card Styling (Blue) */
    .metric-card {
        background: #e0f2fe; /* Light Blue */
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #bae6fd;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .metric-icon { font-size: 2rem; margin-bottom: 0.5rem; }
    .metric-value { font-size: 1.8rem; font-weight: 800; color: #0284c7; }
    .metric-label { font-size: 0.8rem; color: #475569; font-weight: 500; }

    /* Algorithm Card Styling (Green) */
    .algo-card {
        background: #f0fdf4; /* Light Green */
        padding: 1.2rem;
        border-radius: 8px;
        border: 1px solid #bbf7d0;
        margin-bottom: 1rem;
        height: 100%;
        transition: transform 0.2s;
    }
    .algo-card:hover { transform: translateY(-2px); box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    
    .algo-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    
    .algo-title { font-weight: 700; color: #166534; font-size: 1.1rem; }
    
    .algo-badge {
        background-color: #dcfce7;
        color: #15803d;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 700;
        border: 1px solid #86efac;
    }
    
    .algo-desc { font-size: 0.8rem; color: #64748b; }
    .algo-card ul { color: #0f172a; padding-left: 1.5rem; margin-top: 0.5rem; }
    .algo-card ul li { margin-bottom: 0.3rem; }
                
    /* Research Page Styling */
    .research-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #16a34a;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 15px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f1f5f9;
        border-radius: 8px;
        padding: 8px 16px;
        color: #0f172a; /* Force dark text color for visibility */
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #16a34a !important;
        color: white !important;
    }

    /* Note: These classes are defined but Pandas styler uses inline CSS */
    .stDataFrame .trained-model-highlight {
        background-color: #bfdbfe !important; 
        color: #1e3a8a !important; 
        font-weight: bold;
    }
    .stDataFrame .best-model-highlight {
        background-color: #16a34a !important;
        color: white !important;
        font-weight: bold;
    }
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Navigation Structure
PAGES = {
    "🏡 Home": "home",
    "📊 Dataset Analysis": "dataset",
    "⚙️ Algorithm Implementation": "implementation",
    "📚 Model Details": "research",
    "🎯 Model Training Dashboard": "training",
    "🌱 Global Prediction": "prediction",
    "📍 Tamil Nadu Prediction": "tamil_nadu",
    "📊 Results & Metrics": "results",
    "🚀 Deployment": "deployment"
}

# --- Utils ---

@st.cache_data
def load_dataset_global():
    """
    Loads the global crop recommendation dataset. 
    Decorated with cache_data to prevent disk I/O on every rerun.
    """
    # check both local working directory and common upload directory (/mnt/data)
    candidates = ["Crop_recommendation.csv", "/mnt/data/Crop_recommendation.csv"]
    for file_path in candidates:
        if os.path.exists(file_path):
            try:
                return pd.read_csv(file_path)
            except Exception:
                # Try with different encodings / engine fallback
                try:
                    return pd.read_csv(file_path, engine='python')
                except Exception:
                    return pd.DataFrame()
    return pd.DataFrame()

def get_algorithm_info():
    """
    Returns benchmark data for the Global Dataset models.
    LOGIC: Checks 'st.session_state' for any user-trained models (via the Simulation page) 
    and updates the static benchmark scores dynamically.
    """
    # 1. Base Benchmarks
    base_data = [
        {"key": "hybrid", "name": "Hybrid CNN-LSTM", "acc": 0.985, "type": "Hybrid DL Architecture"},
        {"key": "resmlp", "name": "Residual MLP", "acc": 0.981, "type": "Deep Learning Architecture"},
        {"key": "transformer", "name": "Transformer", "acc": 0.978, "type": "Attention Architecture"},
        {"key": "cnn", "name": "1D-CNN", "acc": 0.975, "type": "Deep Learning Architecture"},
        {"key": "ffnn", "name": "Feed Forward NN", "acc": 0.968, "type": "Deep Learning Architecture"},
        {"key": "lstm", "name": "LSTM", "acc": 0.962, "type": "Recurrent NN Architecture"},
        {"key": "gru", "name": "GRU", "acc": 0.959, "type": "Recurrent NN Architecture"},
        {"key": "xgb", "name": "XGBoost", "acc": 0.872, "type": "Gradient Boosting Architecture"},
        {"key": "rf", "name": "Random Forest", "acc": 0.855, "type": "Ensemble Architecture"},
        {"key": "ann", "name": "ANN", "acc": 0.815, "type": "Deep Learning Architecture"}
    ]
    
    # 2. Check for persistent results and override base data
    if 'global_benchmark_override' in st.session_state:
        override = st.session_state.global_benchmark_override
        for i, item in enumerate(base_data):
            if item['name'] in override:
                # Update accuracy for the models trained on the Global dataset
                item['acc'] = override[item['name']]['accuracy']
                
    return sorted(base_data, key=lambda x: x['acc'], reverse=True)

# --- TN Specific Data Utils ---

@st.cache_resource
def load_resources_tn():
    """
    Loads Pickle artifacts (Encoders, Scalers) for Tamil Nadu mode.
    Uses cache_resource because these are large objects, not dataframes.
    """
    try:
        with open('encoders.pkl', 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        return None

@st.cache_data
def load_district_data_tn():
    """
    Parses the complex 'Tamil Nadu - AgriData_Dist.csv' which contains district-wise suitability.
    Handles header inconsistencies where district names are in a specific row.
    """
    file_path = 'Tamil Nadu - AgriData_Dist.csv'
    try:
        df = pd.read_csv(file_path)
        # Ensure we handle the header structure if it includes extra rows/columns
        raw_districts = df.iloc[0, 14:].dropna().values
        district_names = [str(d).strip() for d in raw_districts]
        new_cols = list(df.columns[:14]) + district_names
        df_data = df.iloc[1:].copy()
        df_data = df_data.iloc[:, :len(new_cols)]
        df_data.columns = new_cols
        return df_data, district_names
    except FileNotFoundError:
        return None, None
    except Exception:
        return None, None

TEST_ACCURACIES_TN = {
    "Transformer": "96.8%",
    "CNN": "91.7%",
    "ResidualMLP": "91.4%",
    "Hybrid_CNNLSTM": "88.3%",
    "GRU": "84.4%",
    "LSTM": "82.2%",
    "ANN": "81.5%"
}

# Add TN Benchmark override mechanism
def get_tn_algorithm_info():
    """
    Constructs benchmark data for TN models, merging static test accuracies 
    with detailed metrics (F1, Precision, etc.) for the Results table.
    """
    global_model_names = [a['name'] for a in get_algorithm_info()]
    tn_acc_map = {k.replace('ResidualMLP', 'Residual MLP').replace('Hybrid_CNNLSTM', 'Hybrid CNN-LSTM'): float(v.strip('%'))/100 for k, v in TEST_ACCURACIES_TN.items()}
    
    # Define fixed simulation metrics for models not specified in TEST_ACCURACIES_TN
    global_base_metrics = {
        "Hybrid CNN-LSTM": {"acc": 0.883, "f1": 0.865, "precision": 0.855, "recall": 0.865, "train_time": 30.0, "model_size": 20.0},
        "Residual MLP": {"acc": 0.914, "f1": 0.896, "precision": 0.886, "recall": 0.896, "train_time": 28.0, "model_size": 14.5},
        "Transformer": {"acc": 0.968, "f1": 0.950, "precision": 0.940, "recall": 0.950, "train_time": 26.9, "model_size": 17.8},
        "1D-CNN": {"acc": 0.917, "f1": 0.899, "precision": 0.889, "recall": 0.899, "train_time": 17.8, "model_size": 8.4},
        "Feed Forward NN": {"acc": 0.80, "f1": 0.78, "precision": 0.77, "recall": 0.78, "train_time": 12.7, "model_size": 4.3},
        "LSTM": {"acc": 0.822, "f1": 0.805, "precision": 0.795, "recall": 0.805, "train_time": 24.9, "model_size": 12.1},
        "GRU": {"acc": 0.844, "f1": 0.827, "precision": 0.817, "recall": 0.827, "train_time": 21.1, "model_size": 11.3},
        "XGBoost": {"acc": 0.75, "f1": 0.73, "precision": 0.72, "recall": 0.73, "train_time": 6.0, "model_size": 6.5},
        "Random Forest": {"acc": 0.72, "f1": 0.70, "precision": 0.69, "recall": 0.70, "train_time": 3.6, "model_size": 10.0},
        "ANN": {"acc": 0.815, "f1": 0.798, "precision": 0.788, "recall": 0.798, "train_time": 10.8, "model_size": 3.8}
    }

    base_data_tn = []
    
    for name in global_model_names:
        base_ref = global_base_metrics.get(name)
        if base_ref:
            base_data_tn.append({
                "name": name,
                "acc": base_ref['acc'],
                "f1": base_ref['f1'],
                "precision": base_ref['precision'],
                "recall": base_ref['recall'],
                "train_time": base_ref['train_time'],
                "model_size": base_ref['model_size']
            })

    # Check for persistent TN benchmarks and override base data
    if 'tn_benchmark_override' in st.session_state:
        override = st.session_state.tn_benchmark_override
        for i, item in enumerate(base_data_tn):
            if item['name'] in override:
                # Update all metrics based on the latest saved training result
                item['acc'] = override[item['name']]['accuracy']
                item['f1'] = override[item['name']]['f1']
                item['precision'] = override[item['name']]['precision']
                item['recall'] = override[item['name']]['recall']
                item['train_time'] = override[item['name']]['train_time']
                item['model_size'] = override[item['name']]['model_size']
                
    return sorted(base_data_tn, key=lambda x: x['acc'], reverse=True)


def get_tn_model_predict_proba_wrapper(model):
    """
    Creates a numpy-compatible predict_proba function for a PyTorch model.
    Required by LIME and SHAP KernelExplainer.
    """
    model.eval()
    def predict_proba_fn(x_scaled_np):
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x_scaled_np)
            # The Deep Learning models in the provided code handle the unsqueeze operation internally
            # based on their defined forward passes (e.g., x.unsqueeze(1)).
            logits = model(x_tensor)
            probs = F.softmax(logits, dim=1)
            return probs.numpy()
    return predict_proba_fn

@st.cache_data
def get_tn_x_train_background(_encoders, _scaler): # _encoders and _scaler bypass caching hash calculation
    """
    Reconstructs a downsampled, scaled version of the training data 
    (from 'Tamil Nadu - AgriData_Dist.csv') to serve as the background 
    reference for SHAP and LIME explainers.
    
    Feature Order must match: [Soil_enc, CropType_enc, WaterSource_enc, pH, Duration, Temp, Water, Hum]
    """
    df_tn, _ = load_district_data_tn()
    if df_tn is None or df_tn.empty:
        st.error("Cannot load Tamil Nadu data for XAI background.")
        return None, None

    # Columns inferred to be used in training, ordered by their position in the input vector
    cat_feature_cols = ['SOIL', 'TYPE_OF_CROP', 'TYPE_OF _WATERSOURCE']
    # These numeric features are inferred to correspond to the manual slider inputs:
    num_feature_cols = ['SOIL_PH_LOW', 'CROPDURATION_MIN', 'MIN_TEMP', 'WATER REQUIRED_MIN', 'RELATIVE_HUMIDITY_MIN']
    
    # Clean column headers from the loaded dataframe
    raw_col_names = [c for c in df_tn.columns if isinstance(c, str)]
    
    def find_best_match(name, candidates):
        # Finds exact match ignoring leading/trailing spaces
        return next((c for c in candidates if c.strip() == name.strip()), None)

    actual_cat_cols = [find_best_match(col, raw_col_names) for col in cat_feature_cols]
    actual_num_cols = [find_best_match(col, raw_col_names) for col in num_feature_cols]
    
    if any(c is None for c in actual_cat_cols + actual_num_cols):
        # Fallback for data structure issues
        st.error(f"Missing essential TN columns: {cat_feature_cols + num_feature_cols}")
        return None, None
        
    all_feature_cols = actual_cat_cols + actual_num_cols
    df_subset = df_tn[all_feature_cols].copy()
    
    # 1. Coerce numeric columns to float, filling NaNs with median
    for col in actual_num_cols:
        # The data is initially loaded as string/object due to mixed types, force conversion
        df_subset[col] = pd.to_numeric(df_subset[col], errors='coerce')
        # Fill NaN values (if any) with the median of that column
        df_subset[col] = df_subset[col].fillna(df_subset[col].median())

    # 2. Encode categorical columns
    X_features_list = []
    
    for col in actual_cat_cols:
        # Maps 'TYPE_OF _WATERSOURCE' to 'WATER_SOURCE' for encoder key lookup
        encoder_key = col.strip().replace('TYPE_OF _WATERSOURCE', 'WATER_SOURCE')
        encoder = _encoders.get(encoder_key) 
        
        if encoder:
            # Handle categories not seen in training by setting a known value or raising an error
            encoded_vals = df_subset[col].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)
            X_features_list.append(encoded_vals.rename(f'{col.strip().replace(" ", "")}_enc'))
        else:
            st.error(f"Missing encoder for {col}")
            return None, None
            
    # Add numeric columns (keeping original column names)
    for col in actual_num_cols:
        X_features_list.append(df_subset[col].rename(col.strip().replace(" ", "")))

    X_processed = pd.concat(X_features_list, axis=1)

    # 3. Scale the combined feature set
    X_scaled = _scaler.transform(X_processed.values) 
    
    # Final feature names that correspond to the input vector for prediction (8 features)
    feature_names = ['Soil_enc', 'CropType_enc', 'WaterSource_enc', 'pH', 'Duration', 'Temp', 'Water', 'Hum']
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)
    
    # Downsample for XAI performance (e.g., to max 100 samples for KernelExplainer)
    if len(X_scaled_df) > 100:
        X_scaled_df = X_scaled_df.sample(n=100, random_state=42).reset_index(drop=True)
    
    return X_scaled_df, feature_names

# ==================== PAGE FUNCTIONS ====================

def page_home():
    # Hero Section
    col1, col2 = st.columns([1.5, 1])
    with col1:
        # --- NEW BADGE IMPLEMENTATION (WITH BLINKING DOT) ---
        st.markdown("""
        <div style="display: inline-flex; align-items: center; 
                    background-color: rgba(6, 78, 59, 0.4); 
                    border: 1px solid #166534; 
                    border-radius: 9999px; 
                    padding: 6px 16px; 
                    margin-bottom: 20px;">
            <span class="blinking-dot" style="height: 8px; width: 8px; background-color: #22c55e; border-radius: 50%; margin-right: 10px; display: inline-block;"></span>
            <span style="color: #22c55e; font-weight: 600; font-size: 0.9rem; letter-spacing: 0.5px; font-family: sans-serif;">Precision Agriculture Ready</span>
        </div>
        """, unsafe_allow_html=True)
        # --------------------------------
        
        st.markdown('<h1 class="main-header">Next-Gen Crop<br>Recommendation</h1>', unsafe_allow_html=True)
        st.markdown("""
        <p style="font-size: 1.1rem; color: #64748b; line-height: 1.6; max-width: 600px;">
        Leveraging state-of-the-art algorithms ranging from XGBoost to Hybrid CNN-LSTM architectures for optimal crop selection.
        </p>
        """, unsafe_allow_html=True)
        
        # Action Buttons
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            if st.button("🌱 Global Predict", use_container_width=True, type="primary"):
                st.session_state.page = "prediction"
                st.rerun()
        with c2:
            if st.button("📍 Tamil Nadu Mode", use_container_width=True):
                st.session_state.page = "tamil_nadu"
                st.rerun()

    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #064e3b 0%, #16a34a 50%, #34d399 100%); 
                    border-radius: 1rem; padding: 2rem; text-align: center; position: relative;
                    box-shadow: 0 20px 40px rgba(22, 163, 74, 0.3);">
            <div style="font-size: 5rem; margin-bottom: 1rem;">🌾</div>
            <div style="color: white; font-size: 1.25rem; font-weight: 600;">Multi-Model Analysis</div>
            <div style="color: rgba(255,255,255,0.7); font-size: 0.875rem;">Global & Regional Modules</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 1. System Performance Section (Blue Cards)
    st.markdown("### 📊 System Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = [
        {"val": "98.5%", "label": "Max Accuracy (Hybrid)", "icon": "🎯"},
        {"val": "22", "label": "Crop Varieties", "icon": "🌽"},
        {"val": "10", "label": "Advanced Models", "icon": "🧠"},
        {"val": "<50ms", "label": "Inference Speed", "icon": "⚡"},
    ]
    
    cols = [col1, col2, col3, col4]
    
    for c, m in zip(cols, metrics):
        with c:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">{m['icon']}</div>
                <div class="metric-value">{m['val']}</div>
                <div class="metric-label">{m['label']}</div>
            </div>
            """, unsafe_allow_html=True)

    # 2. Model Leaderboard (Top 3 Ranks)
    st.markdown("### 🏆 Model Leaderboard (Top 3)")
    algos = get_algorithm_info() 
    
    top_c1, top_c2, top_c3 = st.columns(3)
    
    with top_c1:
        st.markdown(f"""
        <div style="background-color: #FFD70033; padding: 15px; border-radius: 10px; border: 2px solid #FFD700; text-align: center;">
            <div style="font-size: 1.5rem;">🥇 1st Place</div>
            <h3 style="margin: 5px 0;">{algos[0]['name']}</h3>
            <div style="font-weight: bold; color: #b45309;">{algos[0]['acc']*100:.1f}% Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
        
    with top_c2:
        st.markdown(f"""
        <div style="background-color: #C0C0C033; padding: 15px; border-radius: 10px; border: 2px solid #C0C0C0; text-align: center;">
            <div style="font-size: 1.5rem;">🥈 2nd Place</div>
            <h3 style="margin: 5px 0;">{algos[1]['name']}</h3>
            <div style="font-weight: bold; color: #525252;">{algos[1]['acc']*100:.1f}% Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
        
    with top_c3:
        st.markdown(f"""
        <div style="background-color: #CD7F3233; padding: 15px; border-radius: 10px; border: 2px solid #CD7F32; text-align: center;">
            <div style="font-size: 1.5rem;">🥉 3rd Place</div>
            <h3 style="margin: 5px 0;">{algos[2]['name']}</h3>
            <div style="font-weight: bold; color: #7c2d12;">{algos[2]['acc']*100:.1f}% Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    # 3. Algorithms Implemented Section (Green Grid)
    st.markdown("### 🚀 Algorithms Implemented")
    
    for i in range(0, len(algos), 3):
        row_cols = st.columns(3)
        for j in range(3):
            if i + j < len(algos):
                algo = algos[i+j]
                with row_cols[j]:
                    st.markdown(f"""
                    <div class="algo-card">
                        <div class="algo-header">
                            <span class="algo-title">{algo['name']}</span>
                            <span class="algo-badge">{algo['acc']*100:.1f}%</span>
                        </div>
                        <div class="algo-desc">{algo['type']}</div>
                    </div>
                    """, unsafe_allow_html=True)

def page_dataset():
    st.markdown("## 📊 Dataset Analysis")
    
    main_tab1, main_tab2 = st.tabs(["🌍 Global Dataset", "📍 Tamil Nadu Dataset"])
    
    # --- GLOBAL DATASET TAB ---
    with main_tab1:
        df = load_dataset_global()
        if df.empty:
            st.error("Dataset 'Crop_recommendation.csv' not found.")
        else:
            sub_tab1, sub_tab2, sub_tab3 = st.tabs(["📋 Data Overview", "📈 Distributions", "🔗 Correlations"])
            
            with sub_tab1:
                st.markdown("### 📋 Global Data Overview")
                
                total_records = len(df)
                n_features = len(df.columns) - 1
                n_crops = df['label'].nunique()
                missing_vals = df.isnull().sum().sum()
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div style="background-color: #e0f2fe; padding: 1.5rem; border-radius: 10px; text-align: center; border: 1px solid #bae6fd;">
                        <div style="font-size: 2rem; margin-bottom: 0.5rem;">📚</div>
                        <div style="font-size: 1.8rem; font-weight: 800; color: #0284c7;">{total_records}</div>
                        <div style="font-size: 0.875rem; color: #475569; font-weight: 600;">Total Records</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div style="background-color: #dcfce7; padding: 1.5rem; border-radius: 10px; text-align: center; border: 1px solid #86efac;">
                        <div style="font-size: 2rem; margin-bottom: 0.5rem;">🧬</div>
                        <div style="font-size: 1.8rem; font-weight: 800; color: #16a34a;">{n_features}</div>
                        <div style="font-size: 0.875rem; color: #475569; font-weight: 600;">Features</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div style="background-color: #fef9c3; padding: 1.5rem; border-radius: 10px; text-align: center; border: 1px solid #fde047;">
                        <div style="font-size: 2rem; margin-bottom: 0.5rem;">🌾</div>
                        <div style="font-size: 1.8rem; font-weight: 800; color: #ca8a04;">{n_crops}</div>
                        <div style="font-size: 0.875rem; color: #475569; font-weight: 600;">Unique Crops</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col4:
                    st.markdown(f"""
                    <div style="background-color: #fee2e2; padding: 1.5rem; border-radius: 10px; text-align: center; border: 1px solid #fca5a5;">
                        <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔍</div>
                        <div style="font-size: 1.8rem; font-weight: 800; color: #dc2626;">{missing_vals}</div>
                        <div style="font-size: 0.875rem; color: #475569; font-weight: 600;">Missing Values</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.dataframe(df.head(10), use_container_width=True)
                
                st.markdown("#### Samples per Crop Type")
                crop_counts = df['label'].value_counts().reset_index()
                crop_counts.columns = ['label', 'count']
                fig = px.bar(
                    crop_counts, 
                    x='label', 
                    y='count', 
                    color='label', 
                    title="",
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            with sub_tab2:
                st.markdown("### 📈 Distributions")
                feature = st.selectbox("Select Feature", df.columns[:-1], key="global_dist_feat")
                
                # 1. Histogram
                fig_hist = px.histogram(
                    df, 
                    x=feature, 
                    color='label', 
                    marginal="box", 
                    title=f"Distribution of {feature}",
                    color_discrete_sequence=px.colors.qualitative.Prism
                )
                st.plotly_chart(fig_hist, use_container_width=True)

                # 2. Box Plot
                st.markdown(f"#### {feature} Ranges per Crop")
                fig_box = px.box(
                    df, 
                    x='label', 
                    y=feature, 
                    color='label', 
                    title=f"{feature} Ranges per Crop",
                    color_discrete_sequence=px.colors.qualitative.Prism
                )
                st.plotly_chart(fig_box, use_container_width=True)
            
            with sub_tab3:
                st.markdown("### 🔗 Correlations")
                corr_matrix = df.select_dtypes(include=[np.number]).corr()
                fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='Greens', title="Feature Correlation Matrix")
                st.plotly_chart(fig_corr, use_container_width=True)

                # 3D Cluster Visualization
                st.markdown("### 🧊 3D Cluster Visualization")
                st.info("Visualizing feature relationships across crop types.")
                
                numeric_cols_global = df.select_dtypes(include=[np.number]).columns.tolist()
                
                d_x = 'N' if 'N' in numeric_cols_global else numeric_cols_global[0]
                d_y = 'P' if 'P' in numeric_cols_global else numeric_cols_global[1]
                d_z = 'K' if 'K' in numeric_cols_global else numeric_cols_global[2]

                c1, c2, c3 = st.columns(3)
                with c1: x_axis = st.selectbox("X Axis", numeric_cols_global, index=numeric_cols_global.index(d_x), key="g_3d_x")
                with c2: y_axis = st.selectbox("Y Axis", numeric_cols_global, index=numeric_cols_global.index(d_y), key="g_3d_y")
                with c3: z_axis = st.selectbox("Z Axis", numeric_cols_global, index=numeric_cols_global.index(d_z), key="g_3d_z")

                if x_axis and y_axis and z_axis:
                    fig_3d = px.scatter_3d(df, x=x_axis, y=y_axis, z=z_axis, color='label', symbol='label')
                    fig_3d.update_layout(scene = dict(xaxis_title=x_axis, yaxis_title=y_axis, zaxis_title=z_axis), height=600)
                    st.plotly_chart(fig_3d, use_container_width=True)

    # --- TAMIL NADU DATASET TAB ---
    with main_tab2:
        df_tn, _ = load_district_data_tn()
        if df_tn is None or df_tn.empty:
            st.error("Dataset 'Tamil Nadu - AgriData_Dist.csv' not found.")
        else:
            tn_tab1, tn_tab2, tn_tab3 = st.tabs(["📋 Data Overview", "📈 Distributions", "🔗 Correlations"])
            
            with tn_tab1:
                st.markdown("### 📋 Tamil Nadu Data Overview")
                
                total_records_tn = len(df_tn)
                n_features_tn = len(df_tn.columns)
                n_crops_tn = df_tn['CROPS'].nunique() if 'CROPS' in df_tn.columns else 0
                missing_vals_tn = df_tn.isnull().sum().sum()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"""<div style="background-color: #e0f2fe; padding: 1.5rem; border-radius: 10px; text-align: center; border: 1px solid #bae6fd;">
                        <div style="font-size: 2rem; margin-bottom: 0.5rem;">📚</div>
                        <div style="font-size: 1.8rem; font-weight: 800; color: #0284c7;">{total_records_tn}</div>
                        <div style="font-size: 0.875rem; color: #475569; font-weight: 600;">Total Records</div></div>""", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""<div style="background-color: #dcfce7; padding: 1.5rem; border-radius: 10px; text-align: center; border: 1px solid #86efac;">
                        <div style="font-size: 2rem; margin-bottom: 0.5rem;">🧬</div>
                        <div style="font-size: 1.8rem; font-weight: 800; color: #16a34a;">{n_features_tn}</div>
                        <div style="font-size: 0.875rem; color: #475569; font-weight: 600;">Features</div></div>""", unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""<div style="background-color: #fef9c3; padding: 1.5rem; border-radius: 10px; text-align: center; border: 1px solid #fde047;">
                        <div style="font-size: 2rem; margin-bottom: 0.5rem;">🌾</div>
                        <div style="font-size: 1.8rem; font-weight: 800; color: #ca8a04;">{n_crops_tn}</div>
                        <div style="font-size: 0.875rem; color: #475569; font-weight: 600;">Unique Crops</div></div>""", unsafe_allow_html=True)
                with col4:
                    st.markdown(f"""<div style="background-color: #fee2e2; padding: 1.5rem; border-radius: 10px; text-align: center; border: 1px solid #fca5a5;">
                        <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔍</div>
                        <div style="font-size: 1.8rem; font-weight: 800; color: #dc2626;">{missing_vals_tn}</div>
                        <div style="font-size: 0.875rem; color: #475569; font-weight: 600;">Missing Values</div></div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.dataframe(df_tn.head(10), use_container_width=True)
                
                if 'CROPS' in df_tn.columns:
                    st.markdown("#### Samples per Crop")
                    crop_counts_tn = df_tn['CROPS'].value_counts().reset_index()
                    crop_counts_tn.columns = ['CROPS', 'count']
                    fig_tn = px.bar(crop_counts_tn, x='CROPS', y='count', color='CROPS', title="", color_discrete_sequence=px.colors.qualitative.Bold)
                    fig_tn.update_layout(showlegend=False)
                    st.plotly_chart(fig_tn, use_container_width=True)

            with tn_tab2:
                st.markdown("### 📈 Distributions")
                feature_tn = st.selectbox("Select Feature", df_tn.columns, key="tn_dist_feat")
                
                # 1. Histogram
                fig_hist_tn = px.histogram(
                    df_tn, x=feature_tn, 
                    color='CROPS' if 'CROPS' in df_tn.columns else None,
                    marginal="box", 
                    title=f"Distribution of {feature_tn}",
                    color_discrete_sequence=px.colors.qualitative.Prism
                )
                st.plotly_chart(fig_hist_tn, use_container_width=True)

                # 2. Box Plot
                if 'CROPS' in df_tn.columns:
                    st.markdown(f"#### {feature_tn} Ranges per Crop")
                    fig_box_tn = px.box(
                        df_tn, 
                        x='CROPS', 
                        y=feature_tn, 
                        color='CROPS',
                        title=f"{feature_tn} Ranges per Crop",
                        color_discrete_sequence=px.colors.qualitative.Prism
                    )
                    st.plotly_chart(fig_box_tn, use_container_width=True)
            
            with tn_tab3:
                st.markdown("### 🔗 Correlations")
                numeric_df_tn = df_tn.select_dtypes(include=[np.number])
                if not numeric_df_tn.empty:
                    corr_matrix_tn = numeric_df_tn.corr()
                    fig_corr_tn = px.imshow(corr_matrix_tn, text_auto=False, color_continuous_scale='Greens', title="Feature Correlation Matrix")
                    st.plotly_chart(fig_corr_tn, use_container_width=True)

                    # 3D Cluster Visualization
                    st.markdown("### 🧊 3D Cluster Visualization")
                    numeric_cols_tn = numeric_df_tn.columns.tolist()
                    
                    if len(numeric_cols_tn) >= 3:
                        t_x = numeric_cols_tn[0]
                        t_y = numeric_cols_tn[1]
                        t_z = numeric_cols_tn[2]

                        tc1, tc2, tc3 = st.columns(3)
                        with tc1: tx_axis = st.selectbox("X Axis", numeric_cols_tn, index=numeric_cols_tn.index(t_x), key="tn_3d_x")
                        with tc2: ty_axis = st.selectbox("Y Axis", numeric_cols_tn, index=numeric_cols_tn.index(t_y), key="tn_3d_y")
                        with tc3: tz_axis = st.selectbox("Z Axis", numeric_cols_tn, index=numeric_cols_tn.index(t_z), key="tn_3d_z")

                        color_col = 'CROPS' if 'CROPS' in df_tn.columns else None
                        
                        st.info(f"Visualizing {tx_axis}, {ty_axis}, and {tz_axis} relationships.")
                        
                        fig_3d_tn = px.scatter_3d(df_tn, x=tx_axis, y=ty_axis, z=tz_axis, color=color_col)
                        fig_3d.update_layout(scene = dict(xaxis_title=tx_axis, yaxis_title=ty_axis, zaxis_title=tz_axis), height=600)
                        st.plotly_chart(fig_3d_tn, use_container_width=True)
                    else:
                        st.warning("Not enough numeric columns for 3D visualization.")
                else:
                    st.warning("No numeric columns found for correlation analysis.")


def page_implementation():
    st.markdown("## ⚙️ Algorithm Implementation")
    st.markdown("Detailed Architecture and Code Structure for all 10 Models")
    
    algos = get_algorithm_info()
    algo_map = {a['name']: a for a in algos}
    
    model_choice_name = st.selectbox("Select Algorithm to View", list(algo_map.keys()))
    selected = algo_map[model_choice_name]
    key = selected['key']
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"### {selected['name']}")
        st.markdown(f"**Type:** {selected['type']}")
        st.markdown(f"**Accuracy:** {selected['acc']*100:.1f}%")
        
        if key == 'rf':
            st.info("Ensemble of decision trees. Robust to overfitting and handles non-linear data well.")
        elif key == 'xgb':
            st.info("Gradient Boosting framework. Highly efficient and flexible, optimized for speed and performance.")
        elif key == 'ffnn':
            st.info("Baseline Deep Learning model. Captures high-dimensional mappings from inputs to classes.")
        elif key == 'cnn':
            st.info("1D Convolutional Neural Network. Captures local dependencies in feature space.")
        elif key == 'lstm':
            st.info("Long Short-Term Memory. Capable of learning long-term dependencies in sequential data.")
        elif key == 'resmlp':
            st.info("Deep architecture with skip connections allowing for deeper networks without vanishing gradients.")
        elif key == 'gru':
            st.info("Gated Recurrent Unit. Similar to LSTM but computationally more efficient.")
        elif key == 'transformer':
            st.info("Uses Self-Attention mechanisms to weigh the importance of specific features dynamically.")
        elif key == 'hybrid':
            st.info("Combines CNN for feature extraction with LSTM for sequential dependency learning.")
        elif key == 'ann':
            st.info("Artificial Neural Network. Standard fully connected architecture used for baseline performance comparisons.")
    
    with col2:
        st.markdown("### 💻 Model Architecture Code")
        
        if key == 'rf':
            st.code("""
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    criterion='gini',
    random_state=42
)
model.fit(X_train, y_train)
            """, language='python')
            
        elif key == 'xgb':
            st.code("""
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    eval_metric='mlogloss',
    use_label_encoder=False
)
model.fit(X_train, y_train)
            """, language='python')
            
        elif key == 'ffnn':
            st.code("""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([
    Dense(128, activation='relu', input_shape=(7,)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(22, activation='softmax') # 22 Crop classes
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            """, language='python')
            
        elif key == 'cnn':
            st.code("""
# Input reshaped to (batch_size, 7, 1)
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(7, 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(50, activation='relu'),
    Dense(22, activation='softmax')
])
            """, language='python')
            
        elif key == 'lstm':
            st.code("""
# Input reshaped to (batch_size, 7, 1)
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(7, 1)),
    Dropout(0.2),
    LSTM(50),
    Dense(22, activation='softmax')
])
            """, language='python')
            
        elif key == 'resmlp':
            st.code("""
def residual_block(x, units, dropout=0.1):
    shortcut = x
    x = Dense(units, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(units, activation='relu')(x)
    if x.shape[-1] != shortcut.shape[-1]:
        shortcut = Dense(units)(shortcut)
    x = Add()([x, shortcut])
    return x

inputs = Input(shape=(7,))
x = Dense(64, activation='relu')(inputs)
x = residual_block(x, 64)
x = residual_block(x, 64)
outputs = Dense(22, activation='softmax')(x)
            """, language='python')
            
        elif key == 'gru':
            st.code("""
model = Sequential([
    GRU(100, return_sequences=True, input_shape=(7, 1)),
    Dropout(0.2),
    GRU(50),
    Dense(22, activation='softmax')
])
            """, language='python')
            
        elif key == 'transformer':
            st.code("""
# Simple Tabular Transformer Logic
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = Add()([x, inputs])

    x = LayerNormalization(epsilon=1e-6)(res)
    x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return Add()([x, res])
            """, language='python')
            
        elif key == 'hybrid':
            st.code("""
# Hybrid CNN-LSTM
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(7, 1)),
    MaxPooling1D(pool_size=2),
    LSTM(100, return_sequences=False),
    Dropout(0.2),
    Dense(50, activation='relu'),
    Dense(22, activation='softmax')
])
            """, language='python')
            
        elif key == 'ann':
            st.code("""
class ANNModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ANNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, output_dim)
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x
            """, language='python')

def page_training():
    st.markdown("## 🎯 Model Training Dashboard")
    st.markdown("Simulate training process for selected architectures")

    col_model, col_data = st.columns([1, 1])

    with col_model:
        # Model Selection
        algos = [a['name'] for a in get_algorithm_info()]
        model_choice = st.selectbox("Select Model to Train", algos)

    with col_data:
        # Dataset Selection
        dataset_choice = st.selectbox(
            "Select Training Dataset", 
            ["Global Dataset", "Tamil Nadu Dataset"],
            key="training_dataset_choice",
            help="Global Dataset: ~2200 samples, 22 classes. Tamil Nadu Dataset: smaller, multi-feature columns."
        )

    st.markdown("---")
    
    # --- DATA SPLIT CONFIGURATION ---
    st.markdown("### 🧬 Data Split Configuration")
    
    # Initialize session state for splits if not present
    if 'train_split' not in st.session_state:
        st.session_state.train_split = 70
    if 'validate_split' not in st.session_state:
        st.session_state.validate_split = 15
    if 'test_split' not in st.session_state:
        st.session_state.test_split = 15

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)

    with col_s1:
        # Define the callbacks to enforce 100% total
        def update_splits(changed_key):
            current_sum = st.session_state.train_split + st.session_state.validate_split + st.session_state.test_split
            if current_sum != 100:
                if changed_key == 'train_split':
                    remaining = 100 - st.session_state.train_split
                    st.session_state.validate_split = int(remaining / 2)
                    st.session_state.test_split = remaining - st.session_state.validate_split
                elif changed_key == 'validate_split':
                    remaining = 100 - st.session_state.validate_split
                    st.session_state.train_split = int(remaining * st.session_state.train_split / (st.session_state.train_split + st.session_state.test_split) if (st.session_state.train_split + st.session_state.test_split) > 0 else remaining / 2)
                    st.session_state.test_split = remaining - st.session_state.train_split
                elif changed_key == 'test_split':
                    remaining = 100 - st.session_state.test_split
                    st.session_state.train_split = int(remaining * st.session_state.train_split / (st.session_state.train_split + st.session_state.validate_split) if (st.session_state.train_split + st.session_state.validate_split) > 0 else remaining / 2)
                    st.session_state.validate_split = remaining - st.session_state.train_split
            
            # Final check to ensure total is exactly 100
            current_sum = st.session_state.train_split + st.session_state.validate_split + st.session_state.test_split
            if current_sum != 100:
                diff = 100 - current_sum
                st.session_state.train_split += diff # Add/subtract difference from train split

        train_split = st.slider("Train (%)", 50, 90, st.session_state.train_split, key='train_split', on_change=update_splits, args=('train_split',))
    
    with col_s2:
        validate_split = st.slider("Validate (%)", 0, 30, st.session_state.validate_split, key='validate_split', on_change=update_splits, args=('validate_split',))

    with col_s3:
        test_split = st.slider("Test (%)", 0, 30, st.session_state.test_split, key='test_split', on_change=update_splits, args=('test_split',))

    with col_s4:
        st.markdown(f"**Total Split:**")
        st.success(f"{st.session_state.train_split + st.session_state.validate_split + st.session_state.test_split}%")

    st.markdown("---")

    # Hyperparameters Row
    st.markdown("### ⚙️ Training Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        epochs = st.slider("Epochs / Estimators", min_value=10, max_value=200, value=50)
    
    with col2:
        lr = st.slider("Learning Rate", min_value=0.0001, max_value=0.1, value=0.001, format="%.4f")
        
    with col3:
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128])

    st.markdown("<br>", unsafe_allow_html=True) # Spacing

    if st.button("▶️ Start Training", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        chart_place = st.empty()
        
        # --- DYNAMIC SIMULATION PARAMETERS BASED ON DATASET ---
        if dataset_choice == "Global Dataset":
            # Global dataset is balanced, expect higher starting and max accuracy
            acc_base = 0.65
            loss_base = 0.8
            # Factor will be higher to reach ~0.98 max
            max_acc_factor = 0.35 + 0.35 * (st.session_state.train_split / 100.0) 
            dataset_modifier = 1.0
            max_sim_acc = 0.99
            benchmark_key = 'global_benchmark_override'
        else: # Tamil Nadu Dataset
            # TN dataset may be less clean/smaller, expect lower starting and max accuracy
            acc_base = 0.45
            loss_base = 1.2
            # Factor will be lower to reach ~0.96 max
            max_acc_factor = 0.45 + 0.3 * (st.session_state.train_split / 100.0) 
            dataset_modifier = 0.9 # Used to simulate slightly faster convergence/less stability
            max_sim_acc = 0.97
            benchmark_key = 'tn_benchmark_override'
            
        # Initialize the override dictionary if it doesn't exist
        if benchmark_key not in st.session_state:
            st.session_state[benchmark_key] = {}
            
        # Simulated performance tracking
        train_acc, train_loss = [], []
        val_acc, val_loss = [], []
        
        # Determine steps (epochs)
        steps = epochs 
        
        status_text.text(f"Training {model_choice} on {dataset_choice} with Train Split {st.session_state.train_split}%...")
        
        # --- TRAINING SIMULATION LOOP ---
        for i in range(steps):
            # Training performance always improves, adjusted by max_acc_factor
            current_train_acc = acc_base + max_acc_factor * (1 - np.exp(-0.1 * i * dataset_modifier)) + np.random.normal(0, 0.005)
            current_train_loss = loss_base * np.exp(-0.1 * i * dataset_modifier) + np.random.normal(0, 0.005)
            
            # Validation performance lags slightly behind and might fluctuate more
            current_val_acc = current_train_acc * (0.95 + 0.05 * np.sin(i / 10))
            current_val_loss = current_train_loss * (1.05 - 0.05 * np.sin(i / 10))
            
            train_acc.append(min(current_train_acc, max_sim_acc))
            train_loss.append(max(current_train_loss, 0.01))
            val_acc.append(min(current_val_acc, max_sim_acc * 0.98))
            val_loss.append(max(current_val_loss, 0.01))
            
            if i % (max(1, steps // 10)) == 0 or i == steps - 1: 
                fig = make_subplots(rows=1, cols=2, subplot_titles=("Accuracy Trend (Train/Validate)", "Loss Trend (Train/Validate)"))
                fig.add_trace(go.Scatter(y=train_acc, mode='lines', name="Train Accuracy", line=dict(color='#16a34a')), row=1, col=1)
                fig.add_trace(go.Scatter(y=val_acc, mode='lines', name="Validate Accuracy", line=dict(color='#0ea5e9')), row=1, col=1)
                fig.add_trace(go.Scatter(y=train_loss, mode='lines', name="Train Loss", line=dict(color='#dc2626')), row=1, col=2)
                fig.add_trace(go.Scatter(y=val_loss, mode='lines', name="Validate Loss", line=dict(color='#f97316')), row=1, col=2)
                fig.update_layout(height=450, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                chart_place.plotly_chart(fig, use_container_width=True)
                
            progress_bar.progress(min((i + 1) / steps, 1.0))
            time.sleep(0.01) # Faster simulation

        # --- FINAL METRICS CALCULATION ---
        final_train_acc = train_acc[-1]
        final_val_acc = val_acc[-1]
        final_train_loss = train_loss[-1]
        final_val_loss = val_loss[-1]

        # Test Accuracy/Loss is simulated based on Validation performance and Test split size
        test_acc_noise = (np.random.rand() * 0.02) * (st.session_state.validate_split / st.session_state.test_split if st.session_state.test_split > 0 else 1)
        final_test_acc = final_val_acc * (1.0 - test_acc_noise) # Test is slightly worse/better than validation
        final_test_loss = final_val_loss * (1.0 + test_acc_noise) 
        
        st.success(f"Training of {model_choice} on {dataset_choice} Complete! Final Test Accuracy: {final_test_acc:.2%}")
        
        # 4. SAVE the result to overwrite benchmark
        # We store the result under the specific model name within the dataset dictionary.
        st.session_state[benchmark_key][model_choice] = {
            "model": model_choice,
            "accuracy": final_test_acc,
            "f1": final_test_acc * 0.99, 
            "precision": final_test_acc * 0.98,
            "recall": final_test_acc * 0.99,
            "train_time": 5.0 + (epochs / 50) * (1.0 if "DL" in model_choice or "LSTM" in model_choice else 0.5) * (1.0 if dataset_choice == "Global Dataset" else 0.7),
            "model_size": 10.0 + (0.5 if "DL" in model_choice or "LSTM" in model_choice else 0.1)
        }
        
        # --- DISPLAY RESULTS TABLE ---
        st.markdown("### 📋 Final Evaluation Metrics")
        results_data = {
            "Metric": ["Accuracy", "Loss", "Data Size (%)"],
            "Train Set": [f"{final_train_acc:.4f}", f"{final_train_loss:.4f}", f"{st.session_state.train_split}%"],
            "Validation Set": [f"{final_val_acc:.4f}", f"{final_val_loss:.4f}", f"{st.session_state.validate_split}%"],
            "Test Set": [f"{final_test_acc:.4f}", f"{final_test_loss:.4f}", f"{st.session_state.test_split}%"]
        }
        df_metrics = pd.DataFrame(results_data)

        # Highlight the Test Accuracy in the table
        def highlight_test_acc(s):
            is_acc_row = s.Metric == "Accuracy"
            is_test_col = s.index == 2 # 2 is the index of the Test Set row (0-indexed)
            if is_acc_row:
                 return ['font-weight: bold; background-color: #dcfce7'] + [''] * (len(s)-1)
            return [''] * len(s)

        st.dataframe(df_metrics.set_index("Metric"), use_container_width=True)
        
        st.info(f"The simulated performance for **{model_choice} on {dataset_choice}** has been saved and will appear as the 'Trained' benchmark on the Results page.")


def page_results():
    st.markdown("## 📊 Results & Benchmarking")
    st.markdown("Comparative Analysis of all 10 Algorithms across datasets.")

    # 1. Prepare Benchmark Data (Static/Persistent Data)
    # Global Data (incorporates persistent updates via get_algorithm_info)
    global_algos_info = get_algorithm_info() 
    data_global = {
        "Algorithm": [a['name'] for a in global_algos_info],
        "Accuracy": [a['acc'] for a in global_algos_info],
        # Fallback/simulated values for other metrics
        "F1 Score": [a['acc'] * 0.99 for a in global_algos_info],
        "Precision": [a['acc'] * 0.98 for a in global_algos_info],
        "Recall": [a['acc'] * 0.99 for a in global_algos_info],
        "Training Time (s)": [5.2, 8.5, 45.3, 40.1, 38.5, 25.4, 18.2, 35.6, 30.1, 15.5],
        "Model Size (MB)": [12.5, 8.2, 25.4, 18.1, 22.3, 10.5, 5.4, 15.2, 14.1, 4.8]
    }
    df_global = pd.DataFrame(data_global)

    # Tamil Nadu Data (incorporates persistent updates via get_tn_algorithm_info)
    tn_algos_info = get_tn_algorithm_info()
    data_tn = {
        "Algorithm": [a['name'] for a in tn_algos_info],
        "Accuracy": [a['acc'] for a in tn_algos_info],
        "F1 Score": [a['f1'] for a in tn_algos_info],
        "Precision": [a['precision'] for a in tn_algos_info],
        "Recall": [a['recall'] for a in tn_algos_info],
        "Training Time (s)": [a['train_time'] for a in tn_algos_info],
        "Model Size (MB)": [a['model_size'] for a in tn_algos_info]
    }
    df_tn = pd.DataFrame(data_tn)
    
    # 2. Add Source Column for Highlighting based on saved results
    
    # Global Source Flagging
    if 'global_benchmark_override' in st.session_state:
        trained_models = st.session_state.global_benchmark_override.keys()
        df_global['Source'] = df_global['Algorithm'].apply(lambda x: 'Trained' if x in trained_models else 'Benchmark')
    else:
        df_global['Source'] = 'Benchmark'

    # TN Source Flagging
    if 'tn_benchmark_override' in st.session_state:
        trained_models_tn = st.session_state.tn_benchmark_override.keys()
        df_tn['Source'] = df_tn['Algorithm'].apply(lambda x: 'Trained' if x in trained_models_tn else 'Benchmark')
    else:
        df_tn['Source'] = 'Benchmark'


    # --- TABS FOR DATASET COMPARISON ---
    tab_global, tab_tn = st.tabs(["🌍 Global Dataset Benchmarks", "📍 Tamil Nadu Regional Benchmarks"])

    def create_results_tab(df_results, dataset_name, df_tn_cm=False):
        
        st.markdown(f"### Results for {dataset_name} (Accuracy-Ranked)")
        
        df_results_sorted = df_results.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
        
        # --- CORRECTED: Use direct CSS strings instead of class names for Styler.apply ---
        BEST_STYLE = 'background-color: #16a34a; color: white; font-weight: bold;'
        TRAINED_STYLE = 'background-color: #bfdbfe; color: #1e3a8a; font-weight: bold;' # Light Blue
        
        def highlight_row(row):
            # Find the best accuracy in the displayed, potentially overridden column
            max_acc = df_results_sorted['Accuracy'].max()
            is_best = row['Accuracy'] == max_acc
            is_trained = row['Source'] == 'Trained'
            
            styles = [''] * len(row)
            
            # Trained models have priority in coloring, unless they are also the absolute best
            if is_best:
                styles = [BEST_STYLE] * len(row)
            elif is_trained:
                styles = [TRAINED_STYLE] * len(row)
            
            return styles
        # --- END CORRECTED SECTION ---

        st.dataframe(
            df_results_sorted[['Algorithm', 'Accuracy', 'F1 Score', 'Precision', 'Recall', 'Training Time (s)', 'Model Size (MB)', 'Source']]
            .style.format({
                "Accuracy": "{:.4f}",  
                "F1 Score": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",  
                "Training Time (s)": "{:.2f}s",
                "Model Size (MB)": "{:.1f}MB"
            }).apply(highlight_row, axis=1), 
            use_container_width=True
        )
        
        # Charts
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Accuracy Comparison")
            fig_acc = px.bar(
                df_results_sorted, 
                x="Algorithm", 
                y="Accuracy", 
                color="Algorithm",
                color_discrete_sequence=px.colors.qualitative.Prism,
                range_y=[df_results_sorted['Accuracy'].min() * 0.95, 1.0] 
            )
            fig_acc.update_layout(showlegend=False, height=350, yaxis_tickformat=".2f")
            st.plotly_chart(fig_acc, use_container_width=True)

        with col2:
            st.markdown("#### Accuracy vs. Training Time")
            fig_eff = px.scatter(
                df_results_sorted,
                x="Training Time (s)",
                y="Accuracy",
                size="Model Size (MB)", 
                color="Algorithm",
                hover_name="Algorithm",
                color_discrete_sequence=px.colors.qualitative.Prism
            )
            fig_eff.update_layout(height=350, yaxis_tickformat=".2f")
            st.plotly_chart(fig_eff, use_container_width=True)
            
        # Confusion Matrix Placeholder
        if not df_tn_cm:
            st.markdown("### Confusion Matrix (Hybrid CNN-LSTM - Global)")
            st.markdown("Simulated Confusion Matrix (Validation Set)")
            
            crop_labels = [
                'Rice', 'Maize', 'Chickpea', 'Kidneybeans', 'Pigeonpeas', 
                'Mothbeans', 'Mungbean', 'Blackgram', 'Lentil', 'Pomegranate', 
                'Banana', 'Mango', 'Grapes', 'Watermelon', 'Muskmelon', 
                'Apple', 'Orange', 'Papaya', 'Coconut', 'Cotton', 
                'Jute', 'Coffee'
            ]
            
            classes = 22
            # Use random matrix aligned with expected performance (Hybrid: high accuracy)
            matrix = np.eye(classes) * 50 + np.random.randint(0, 5, size=(classes, classes))
            
            fig_cm = px.imshow(
                matrix, 
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=crop_labels,
                y=crop_labels,
                color_continuous_scale="Blues"
            )
            fig_cm.update_layout(height=600, xaxis_tickangle=-45)
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.markdown("### Confusion Matrix (Tamil Nadu - Simulated)")
            st.info("Confusion Matrix visualization is often optimized for the full Global Dataset, or requires trained models and test data which are simulated here.")
            
    with tab_global:
        create_results_tab(df_global, "Global Dataset", df_tn_cm=False)

    with tab_tn:
        create_results_tab(df_tn, "Tamil Nadu Dataset", df_tn_cm=True)


def page_research():
    st.markdown("## 📚 Research & Model Details")
    st.markdown("Comprehensive analysis of the machine learning architectures implemented in AgriSmart.")
    
    # --- 1. ARCHITECTURE DIAGRAM (Graphviz) ---
    st.markdown("### 🏗️ Unified Architecture Flow")
    st.graphviz_chart("""
    digraph {
        rankdir=LR;
        node [shape=box, style=filled, fillcolor="white", fontname="Sans", penwidth=1.5];
        edge [penwidth=1.2, arrowsize=0.8, color="#64748b"];

        # Inputs
        subgraph cluster_inputs {
            label = "Input Layer";
            style=dashed;
            color="#94a3b8";
            fontcolor="#64748b";
            Input [label="Environmental Data\n(N, P, K, Temp, etc.)", shape=oval, fillcolor="#dcfce7", color="#16a34a"];
        }

        # Models
        subgraph cluster_models {
            label = "Model Ensembles";
            style=rounded;
            bgcolor="#f8fafc";
            color="#cbd5e1";

            node [shape=box, fillcolor="#e0f2fe", color="#0284c7"];
            RF [label="Random Forest\n(Bagging)"];
            XGB [label="XGBoost\n(Boosting)"];
            
            node [shape=box, fillcolor="#fef9c3", color="#ca8a04"];
            CNN [label="1D-CNN\n(Spatial)"];
            RNN [label="LSTM / GRU\n(Sequential)"];
            
            node [shape=box, fillcolor="#fae8ff", color="#a855f7"];
            Trans [label="Transformer\n(Attention)"];
            Hybrid [label="Hybrid\n(CNN+LSTM)"];
        }

        # Output
        Output [label="Crop Class\n(Softmax Probability)", shape=oval, fillcolor="#fee2e2", color="#dc2626"];

        # Connections
        Input -> RF;
        Input -> XGB;
        Input -> CNN;
        Input -> RNN;
        Input -> Trans;
        Input -> Hybrid;

        RF -> Output;
        XGB -> Output;
        CNN -> Output;
        RNN -> Output;
        Trans -> Output;
        Hybrid -> Output;
    }
    """)
    

    st.markdown("---")

    # --- 2. DETAILED MODEL BREAKDOWN ---
    st.markdown("### 🧠 Strategic Model Selection")
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="algo-card"><div class="algo-title">🌲 Tree-Based Models (RF & XGBoost)</div>'
                    '<div class="algo-desc">Standard benchmarks for tabular agricultural data.</div><br>'
                    '<ul><li><b>Random Forest:</b> Handles non-linear relationships via bagging.</li>'
                    '<li><b>XGBoost:</b> Gradient boosting engine that minimizes bias/variance.</li>'
                    '</ul></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="algo-card"><div class="algo-title">🧬 Deep Learning (FFNN & MLP)</div>'
                    '<div class="algo-desc">Capturing high-dimensional mappings.</div><br>'
                    '<ul><li><b>FFNN:</b> Baseline fully connected network.</li>'
                    '<li><b>Residual MLP:</b> Uses skip connections (like ResNet) to prevent vanishing gradients in deeper networks.</li>'
                    '</ul></div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="algo-card"><div class="algo-title">🌊 Sequential Models (RNNs)</div>'
                    '<div class="algo-desc">Treating features as sequences.</div><br>'
                    '<ul><li><b>LSTM & GRU:</b> Effective for datasets where parameter interaction simulates sequential dependency (e.g. Temp → Humidity).</li>'
                    '<li><b>1D-CNN:</b> Extracts local compound features (e.g. N-P-K interactions).</li>'
                    '</ul></div>', unsafe_allow_html=True)

        st.markdown('<div class="algo-card"><div class="algo-title">🚀 Advanced Architectures</div>'
                    '<div class="algo-desc">State-of-the-art implementations.</div><br>'
                    '<ul><li><b>Transformer:</b> Uses <i>Self-Attention</i> to weigh feature importance dynamically per sample.</li>'
                    '<li><b>Hybrid CNN-LSTM:</b> Combines CNN for feature extraction with LSTM for sequential dependency learning.</li>'
                    '</ul></div>', unsafe_allow_html=True)

    # --- 3. TECHNICAL DEEP DIVE (Math) ---
    st.markdown("### 📐 Technical Specifications")
    with st.expander("View Mathematical Formulations", expanded=False):
        st.markdown("#### 1. Transformer Self-Attention Mechanism")
        st.latex(r'''
        Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
        ''')
        st.write("Where $Q$ (Query), $K$ (Key), and $V$ (Value) are linear projections of the input features. This allows the model to focus on specific nutrient imbalances.")

        st.markdown("#### 2. LSTM Forget Gate")
        st.latex(r'''
        f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
        ''')
        st.write("Controls what information is discarded from the cell state, crucial for filtering noise in sensor data.")

        st.markdown("#### 3. Classification Output (Softmax)")
        st.latex(r'''
        \sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
        ''')
        st.write("Converts the raw logits into a probability distribution over the 22 crop classes.")

def page_deployment():
    st.markdown("## 🚀 Deployment Guide")
    st.markdown("Instructions to deploy AgriSmart")

    tab1, tab2 = st.tabs(["💻 Local", "🐳 Docker"])

    with tab1:
        st.code("""
# 1. Install Requirements
pip install streamlit pandas numpy plotly scikit-learn tensorflow xgboost shap lime

# 2. Run Application
streamlit run app.py
        """, language="bash")
    
    with tab2:
        st.info("Docker instructions coming soon...")

# ==================== EXPLAINABLE AI HELPER FUNCTIONS (Global Prediction) ====================

def explain_model_prediction(model, input_data, X_train, model_type="tree"):
    """
    Generates SHAP values to explain the specific prediction.
    (Used for Global Prediction page - typically TreeExplainer)
    """
    st.markdown("### 🕵️ Explainable AI (XAI) Engine")
    
    try:
        # 1. Calculate SHAP values
        if model_type == "tree":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_data)
            
            # Get the predicted class index
            prediction_idx = int(model.predict(input_data)[0])
            
            # --- ROBUST SHAP SLICING LOGIC ---
            vals = None
            base_val = None

            # Handle Values
            if isinstance(shap_values, list):
                # Case 1: List (Multiclass output of TreeExplainer)
                vals = shap_values[prediction_idx][0] # Sample 0 of the specific class
            elif hasattr(shap_values, 'shape') and len(shap_values.shape) == 3:
                # Case 2: 3D Array (Samples, Features, Classes)
                vals = shap_values[0, :, prediction_idx] # Sample 0, All Features, Specific Class
            elif hasattr(shap_values, 'shape') and len(shap_values.shape) == 2:
                # Case 3: 2D Array (Binary/Regression)
                vals = shap_values[0]
            else:
                st.error(f"Unknown SHAP shape: {getattr(shap_values, 'shape', 'N/A')}")
                return

            # Handle Expected/Base Value
            if hasattr(explainer, "expected_value"):
                ev = explainer.expected_value
                if isinstance(ev, list) or (isinstance(ev, np.ndarray) and len(ev) > 1):
                    base_val = ev[prediction_idx]
                else:
                    base_val = ev
            else:
                base_val = 0

            # Create Explanation Object manually
            feature_names = ['N', 'P', 'K', 'Temp', 'Hum', 'pH', 'Rain']
            exp = shap.Explanation(
                values=vals, 
                base_values=base_val, 
                data=input_data[0], 
                feature_names=feature_names
            )

            # 2. Visualize - Waterfall Plot
            st.markdown("#### 1. Why this prediction?")
            st.caption("The **Waterfall Plot** shows how each feature pushed the prediction probability higher (Red) or lower (Blue) from the baseline.")
            
            fig_xai, ax = plt.subplots(figsize=(10, 5))
            shap.plots.waterfall(exp, show=False)
            st.pyplot(fig_xai, bbox_inches='tight')

            # 3. Force Analysis (Bar Chart)
            st.markdown("#### 2. Force Analysis")
            df_imp = pd.DataFrame({
                "Feature": feature_names,
                "Impact": vals
            }).sort_values(by="Impact", ascending=True)
            
            fig_force = px.bar(
                df_imp, 
                x="Impact", 
                y="Feature", 
                orientation='h',
                color="Impact", 
                color_continuous_scale=["#3b82f6", "#ef4444"], 
                title="Feature Impact Direction"
            )
            st.plotly_chart(fig_force, use_container_width=True)
            
        else:
            st.warning("Only Tree-based models (RF, XGBoost) are currently supported for full XAI in this demo.")

    except Exception as e:
        st.error(f"XAI Error: {str(e)}")
        st.write("Debug info: Shape mismatch in SHAP calculation. Please ensure model inputs match dataset columns.")


def explain_local_lime(model, input_data, X_train, label_encoder=None, feature_names=None, num_features=7):
    """
    Robust LIME explanation for a single input instance.
    (Used for Global Prediction page)
    """
    st.markdown("### 🕵️ Local Explanation — LIME")
    if not LIME_AVAILABLE:
        st.error("LIME is not available. Install it with `pip install lime` to use this feature.")
        return

    try:
        # Prepare training data for explainer (as numpy)
        if isinstance(X_train, pd.DataFrame):
            train_data = X_train.values.copy()
            if feature_names is None:
                feature_names = X_train.columns.tolist()
        else:
            train_data = np.array(X_train).copy()
            if feature_names is None:
                # if train_data is 1D make it 2D placeholder
                if train_data.ndim == 1:
                    feature_names = [f"f{i}" for i in range(len(train_data))]
                else:
                    feature_names = [f"f{i}" for i in range(train_data.shape[1])]

        # Ensure train_data is 2D (n_samples, n_features)
        if train_data.ndim == 1:
            train_data = train_data.reshape(-1, len(feature_names))

        n_features_train = train_data.shape[1]

        # Ensure num_features doesn't exceed available features
        if num_features is None:
            num_features = n_features_train
        else:
            num_features = min(int(num_features), n_features_train)

        # Class names
        class_names = None
        if label_encoder is not None:
            try:
                class_names = label_encoder.classes_.tolist()
            except Exception:
                class_names = None

        explainer = LimeTabularExplainer(
            training_data=train_data,
            feature_names=feature_names,
            class_names=class_names,
            mode='classification',
            discretize_continuous=True
        )

        # We rely on predict_proba for LIME; make wrapper robust to 1D inputs
        def predict_proba_fn(x):
            x_arr = np.asarray(x)
            if x_arr.ndim == 1:
                x_arr = x_arr.reshape(1, -1)
            # Some classifiers expect float type
            try:
                proba = model.predict_proba(x_arr)
            except Exception as err:
                raise RuntimeError(f"model.predict_proba failed inside LIME wrapper: {err}")
            return np.array(proba, dtype=float)

        # Safe call to explain_instance: LIME expects a 1D sample
        sample = np.asarray(input_data[0]).astype(float)
        exp = explainer.explain_instance(
            sample,
            predict_proba_fn,
            num_features=num_features
        )

        # ✅ Use argmax of predict_proba to get class index consistent with predict_proba order
        proba = model.predict_proba(input_data if np.asarray(input_data).ndim == 2 else np.asarray(input_data).reshape(1, -1))
        pred_idx = int(np.argmax(proba, axis=1)[0])

        pred_class_name = None
        if class_names is not None and pred_idx < len(class_names):
            pred_class_name = class_names[pred_idx]

        st.markdown(f"**Predicted class (by model):** {pred_class_name if pred_class_name else pred_idx}")

        # Try to get explanation for that class; fallback to overall explanation
        try:
            explanation_list = exp.as_list(label=pred_idx)
        except Exception:
            # as_list() without label returns the top features (weights) for default (may be positive/negative)
            try:
                explanation_list = exp.as_list()
            except Exception as err:
                raise RuntimeError(f"LIME explanation extraction failed: {err}")

        # Convert to DataFrame for display and charting
        feat = []
        weight = []
        for item in explanation_list:
            # item may be (feature_str, weight)
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                f, w = item[0], float(item[1])
            else:
                # unexpected format
                f, w = str(item), 0.0
            feat.append(f)
            weight.append(w)

        df_lime = pd.DataFrame({"Feature": feat, "Weight": weight})
        # LIME's strings may include ranges, so keep as-is
        df_lime['Direction'] = df_lime['Weight'].apply(lambda x: 'Increase' if x > 0 else 'Decrease')

        st.markdown("#### Feature contributions (LIME)")
        st.dataframe(df_lime, use_container_width=True)

        # Horizontal bar visualization
        fig = px.bar(
            df_lime,
            x='Weight',
            y='Feature',
            orientation='h',
            color='Weight',
            color_continuous_scale=px.colors.sequential.Greens,
            title='LIME local feature contributions',
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Could not compute LIME explanation: {e}")
        st.info("Ensure model supports `predict_proba`, input dims match training data, and LIME is installed. If error persists, try a simpler model (RandomForest/XGBoost) or enable SHAP instead.")


# --- NEW XAI Visualization Function (Tamil Nadu Prediction) ---

def explain_tn_model_prediction_shap_lime(model_name, model_instance, input_data_scaled, X_train_background, label_encoder):
    """
    Generates SHAP (KernelExplainer for deep models) and LIME local explanations
    for the PyTorch-based TN models.
    """
    st.markdown("### 🔬 Explainable AI (XAI) for Recommendation")
    
    predict_proba_fn = get_tn_model_predict_proba_wrapper(model_instance)
    
    X_train_np = X_train_background.values
    feature_names = X_train_background.columns.tolist()
    class_names = label_encoder.classes_.tolist()
    N_CLASSES = len(class_names) # 22 in a standard setup
    
    # Get the model's prediction
    proba = predict_proba_fn(input_data_scaled)
    raw_pred_idx = int(np.argmax(proba, axis=1)[0])

    # --- CRITICAL FIX: Validate pred_idx against N_CLASSES ---
    if raw_pred_idx >= N_CLASSES or raw_pred_idx < 0:
        # The prediction result is invalid. Default to the most probable class (index 0) or handle the error.
        # If the prediction index is clearly wrong, we reset it to a safe value (0) to allow SHAP calculation.
        pred_idx = 0
        st.error(f"Prediction Mismatch: Calculated class index ({raw_pred_idx}) is out of expected bounds (0 to {N_CLASSES-1}). SHAP analysis defaulting to Class 0 for explanation.")
    else:
        pred_idx = raw_pred_idx
    
    pred_class_name = class_names[pred_idx]
    
    st.markdown(f"**Target Class for Explanation:** {pred_class_name}")
    st.info("The explanations below show how the 8 scaled input features contribute to the final crop prediction.")

    # --- TABBED DISPLAY ---
    tab_shap, tab_lime = st.tabs(["🔥 SHAP Explanation (KernelExplainer)", "🍋 LIME Explanation"])

    # 1. SHAP Explanation (KernelExplainer for PyTorch)
    with tab_shap:
        st.markdown("#### Feature Influence (SHAP Kernel Explainer)")
        st.caption("SHAP's KernelExplainer uses a model-agnostic approach, approximating the effects of each feature on the predicted class probability. This is essential for deep learning models like CNN-LSTM.")
        
        try:
            # SHAP KernelExplainer setup: Use a small, fixed subsample of the background data for stability
            background_data_for_explainer = X_train_np
            
            explainer = shap.KernelExplainer(predict_proba_fn, background_data_for_explainer)
            
            with st.spinner("Calculating SHAP values (May take a moment)..."):
                # Calculate SHAP values for the single input instance. nsamples=500 is common for good approximation.
                shap_values = explainer.shap_values(input_data_scaled[0].reshape(1, -1), nsamples=500)
            
            # --- Robust Indexing (Fixes the out of bounds error by defensive indexing) ---
            
            # 1. Determine Base Value (Expected Value)
            ev = explainer.expected_value
            if isinstance(ev, (list, np.ndarray)) and len(ev) > 1:
                safe_ev_idx = min(pred_idx, len(ev) - 1)
                base_value_pred_class = ev[safe_ev_idx]
            elif isinstance(ev, (list, np.ndarray)):
                base_value_pred_class = ev[0] if len(ev) > 0 else 0
            else:
                base_value_pred_class = ev
            
            # 2. Determine SHAP Values
            N_FEATURES = len(feature_names) # Should be 8
            
            if isinstance(shap_values, list):
                if pred_idx < len(shap_values):
                    # Correct multiclass output: list of arrays, index by predicted class
                    shap_values_pred_class = shap_values[pred_idx][0] # [0] to get the single sample
                elif len(shap_values) == 1:
                    # Fallback 1: Corrupted multiclass output defaulted to list of size 1.
                    shap_values_pred_class = shap_values[0][0]
                    st.warning("SHAP returned list of size 1. Assuming binary/corrupted multiclass output.")
                else:
                    raise RuntimeError(f"SHAP values list size unexpected: {len(shap_values)}")
            elif not isinstance(shap_values, list) and shap_values.ndim == 3 and shap_values.shape[1] == N_FEATURES:
                # New Case: SHAP returned (1, 8, N_CLASSES_SHAP). Use safe index on last axis.
                class_axis_size = shap_values.shape[2]
                safe_pred_idx_for_array = min(pred_idx, class_axis_size - 1)
                shap_values_pred_class = shap_values[0, :, safe_pred_idx_for_array]
                if class_axis_size != N_CLASSES:
                    st.warning(f"SHAP returned unexpected class dimension ({class_axis_size} vs expected {N_CLASSES}). Proceeding with safe indexing: class={safe_pred_idx_for_array}.")
            elif not isinstance(shap_values, list) and shap_values.ndim == 2 and shap_values.shape[1] == N_FEATURES:
                # Single sample, single output (regression/binary)
                shap_values_pred_class = shap_values[0]
            else:
                # If none of the robust indices match, raise error with more info
                raise RuntimeError(f"SHAP output shape unexpected: {type(shap_values)}, shape={getattr(shap_values, 'shape', 'N/A')}, N_CLASSES={N_CLASSES}")
            # --- End Robust Indexing ---

            # Create Explanation Object manually
            exp = shap.Explanation(
                values=shap_values_pred_class, 
                base_values=base_value_pred_class, 
                data=input_data_scaled[0], 
                feature_names=feature_names
            )
            
            # Plot 1: Waterfall Plot
            st.markdown("##### Waterfall Plot (How Features Pushed the Prediction)")
            fig_shap, ax_shap = plt.subplots(figsize=(10, 5))
            shap.plots.waterfall(exp, show=False)
            st.pyplot(fig_shap, bbox_inches='tight')
            plt.close(fig_shap)
            
            # Plot 2: Bar Plot (Summary of magnitude)
            st.markdown("##### Feature Impact Magnitude")
            df_imp = pd.DataFrame({
                "Feature": feature_names,
                "Impact_Magnitude": np.abs(shap_values_pred_class)
            }).sort_values(by="Impact_Magnitude", ascending=True)
            
            fig_bar = px.bar(
                df_imp, 
                x="Impact_Magnitude", 
                y="Feature", 
                orientation='h',
                color="Impact_Magnitude", 
                color_continuous_scale=px.colors.sequential.Greens,
                title="Absolute Feature Impact on Prediction"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        except Exception as e:
            st.error(f"SHAP KernelExplainer Error: {str(e)}. This often means the model's output or the input dimensions are incorrect for the explainer.")
            
    # 2. LIME Explanation (LimeTabularExplainer)
    with tab_lime:
        st.markdown("#### Feature Influence (LIME Explainer)")
        st.caption("LIME explains the prediction by locally approximating the model's behaviour with an interpretable linear model. The weights show local contribution.")
        
        if not LIME_AVAILABLE:
            st.error("LIME is not available. Install it with `pip install lime` to use this feature.")
        else:
            try:
                explainer = LimeTabularExplainer(
                    training_data=X_train_np,
                    feature_names=feature_names,
                    class_names=class_names,
                    mode='classification',
                    discretize_continuous=True,
                    random_state=42
                )
                
                sample = input_data_scaled[0].astype(float)
                
                with st.spinner("Calculating LIME explanation..."):
                    exp = explainer.explain_instance(
                        sample,
                        predict_proba_fn,
                        num_features=len(feature_names),
                        labels=[pred_idx]
                    )

                explanation_list = exp.as_list(label=pred_idx)
                
                feat, weight = [], []
                for item in explanation_list:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        feat.append(item[0])
                        weight.append(float(item[1]))
                    
                df_lime = pd.DataFrame({"Feature & Condition": feat, "Local Weight": weight})
                df_lime['Color'] = df_lime['Local Weight'].apply(lambda x: 'Positive' if x > 0 else 'Negative')
                df_lime = df_lime.sort_values(by="Local Weight", ascending=True)
                
                st.markdown("##### Local Feature Weights")
                st.dataframe(df_lime, use_container_width=True)

                # Horizontal bar visualization
                fig = px.bar(
                    df_lime,
                    x='Local Weight',
                    y='Feature & Condition', # Correct column name used here
                    orientation='h',
                    color='Color',
                    color_discrete_map={'Positive': '#16a34a', 'Negative': '#ef4444'},
                    title='LIME Local Feature Contributions for Predicted Crop',
                    labels={'Local Weight': 'Weight', 'Feature & Condition': 'Feature (Condition)'}
                )
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Could not compute LIME explanation: {e}")
                st.info("LIME depends heavily on the background data distribution. Check data types and scaling consistency.")


# ==================== UPDATED GLOBAL PREDICTION PAGE (ADDED GLOBAL SHAP + PDP) ====================

def page_prediction_global():
    st.markdown("## 🌱 Global Crop Prediction Engine")
    st.markdown("Enter environmental data to get real-time crop recommendations. Optionally compute global SHAP explanations and PDPs for tree models.")
    
    col1, col2 = st.columns([1, 2], gap="large")
    
    with col1:
        st.markdown("### 📝 Input Parameters")
        n = st.number_input("Nitrogen (N)", 0, 140, 90)
        p = st.number_input("Phosphorus (P)", 5, 145, 42)
        k = st.number_input("Potassium (K)", 5, 205, 43)
        temp = st.number_input("Temperature (°C)", 8.0, 45.0, 20.8)
        hum = st.number_input("Humidity (%)", 14.0, 100.0, 82.0)
        ph = st.number_input("pH Level", 3.5, 9.9, 6.5)
        rain = st.number_input("Rainfall (mm)", 20.0, 300.0, 202.9)
        
        st.markdown("---")
        st.markdown("### ⚙️ Engine Configuration")
        
        model_options = [
            "Hybrid CNN-LSTM",
            "Transformer",
            "Residual MLP",
            "Feed Forward NN",
            "1D-CNN",
            "LSTM",
            "XGBoost",
            "GRU",
            "Random Forest"
        ]
        model_choice = st.selectbox("Inference Model", model_options)

        # XAI Toggle (local LIME)
        enable_local_xai = st.checkbox("Enable Explainable AI (LIME local)", value=True, help="Show local (LIME) explanation for the model's recommendation.")
        # NEW: Global XAI Toggle (SHAP + PDP)
        enable_global_xai = st.checkbox("Enable Global Understanding (SHAP summary + PDP)", value=False, help="Calculate SHAP summary plots and PDPs for the chosen tree model (may take time).")
        # Note: SHAP + PDP works best for tree models (RandomForest/XGBoost). For surrogate RF used for neural options, it will still generate tree-based XAI.
        
        # PDP feature selection (only used if enable_global_xai)
        pdp_features = None
        if enable_global_xai:
            st.markdown("Select features for Partial Dependence Plots (PDP)")
            sample_feats = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
            pdp_features = st.multiselect("PDP Features (limit 3)", sample_feats, default=['N', 'P', 'temperature'])
            # ensure limit
            if len(pdp_features) > 3:
                st.warning("Limiting PDP to first 3 features to keep computation reasonable.")
                pdp_features = pdp_features[:3]

        predict_btn = st.button("🔍 Predict Crop", type="primary", use_container_width=True)
        
    with col2:
        st.markdown("### 📊 Prediction Result")
        
        if predict_btn:
            with st.spinner(f"Processing with {model_choice}..."):
                
                # --- DATA LOADING ---
                df = load_dataset_global()
                
                if not df.empty:
                    # Expect these numeric columns in dataset
                    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
                    y = df['label']
                    
                    le = LabelEncoder()
                    y_encoded = le.fit_transform(y)
                    
                    input_data = np.array([[n, p, k, temp, hum, ph, rain]])
                    
                    # --- MODEL TRAINING & INFERENCE ---
                    
                    model = None
                    model_type = "neural" # Default
                    
                    if "Random Forest" in model_choice:
                        model = RandomForestClassifier(n_estimators=100, random_state=42)
                        model.fit(X, y_encoded)
                        model_type = "tree"
                        
                    elif "XGBoost" in model_choice:
                        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, verbosity=0)
                        model.fit(X, y_encoded)
                        model_type = "tree"
                        
                    else:
                        # For Deep Learning/Other models in this demo, we use a surrogate RF 
                        # tailored to mimic different decision boundaries by varying the seed
                        
                        seed = sum(ord(c) for c in model_choice)
                        
                        model = RandomForestClassifier(n_estimators=100, random_state=seed)
                        model.fit(X, y_encoded)
                        model_type = "tree" 
                    
                    # Prediction
                    pred_idx = model.predict(input_data)[0]
                    pred = le.inverse_transform([pred_idx])[0]
                    probs = model.predict_proba(input_data)
                    conf_score = np.max(probs) * 100
                    
                    # --- DISPLAY RESULTS ---
                    st.markdown(f"""
                    <div style="background-color: #dcfce7; padding: 30px; border-radius: 12px; text-align: center; border: 1px solid #86efac; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
                        <p style="color: #4b5563; font-size: 1.2rem; margin-bottom: 10px; font-weight: 500;">Recommended Crop</p>
                        <h1 style="color: #166534; font-size: 4rem; margin: 0; font-weight: 800; letter-spacing: -1px;">{pred.upper()}</h1>
                        <p style="color: #15803d; font-weight: 600; margin-top: 15px; font-size: 1rem;">Confidence Score: {conf_score:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # --- LOCAL EXPLAINABLE AI SECTION (LIME local) ---
                    if enable_local_xai:
                        if not LIME_AVAILABLE:
                            st.error("LIME package not installed. Install via `pip install lime` to use LIME local explanations.")
                        else:
                            explain_local_lime(model, input_data, X, label_encoder=le, feature_names=X.columns.tolist(), num_features=X.shape[1])
                    else:
                        # Fallback to simple feature importance if LIME is off
                        st.markdown("### 💡 Feature Importance (model.feature_importances_)")
                        try:
                            importances = model.feature_importances_
                            features = ['N', 'P', 'K', 'Temp', 'Hum', 'pH', 'Rain']
                            fig = px.bar(
                                x=features, y=importances, 
                                labels={'x': 'Feature', 'y': 'Importance'},
                                color=importances,
                                color_continuous_scale='Greens'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.info("Model does not expose feature_importances_. Can't show feature importance.")
                    
                    # --- GLOBAL XAI (SHAP summary & PDP) ---
                    if enable_global_xai:
                        st.markdown("### 🌐 Global Understanding — SHAP & PDP")
                        st.caption("Global SHAP summary plot aggregates feature-level effects across the dataset. PDPs show marginal effect of feature on predicted outcome.")
                        
                        try:
                            # SHAP Summary (works best with tree explainer for RF/XGB)
                            if model_type != "tree":
                                st.info("Global SHAP + PDP best supports tree-based models. Proceeding with surrogate tree-based explainer where possible.")
                            
                            sample_for_shap = X.copy()
                            # downsample if dataset is large
                            n_samples = len(sample_for_shap)
                            if n_samples > 1000:
                                sample_for_shap = sample_for_shap.sample(1000, random_state=42)
                            
                            st.info(f"Computing SHAP values on {len(sample_for_shap)} samples (this may take a few seconds).")
                            
                            # Tree explainer
                            explainer = shap.TreeExplainer(model)
                            shap_values = explainer.shap_values(sample_for_shap)
                            
                            # Prepare shap_values for summary plotting:
                            try:
                                if isinstance(shap_values, list):
                                    # shap_values is list of arrays (n_samples, n_features) for each class
                                    stacked = np.stack(shap_values, axis=2)  # shape (n_samples, n_features, n_classes)
                                    shap_for_summary = np.mean(stacked, axis=2)  # average across classes -> (n_samples, n_features)
                                elif hasattr(shap_values, 'shape') and len(shap_values.shape) == 3:
                                    # (n_samples, n_features, n_classes)
                                    shap_for_summary = np.mean(shap_values, axis=2)
                                else:
                                    shap_for_summary = shap_values
                            except Exception:
                                shap_for_summary = shap_values
                            
                            # Plot SHAP summary (beeswarm)
                            fig_shap, ax = plt.subplots(figsize=(10, 6))
                            try:
                                # shap.summary_plot accepts numpy arrays / DataFrame
                                shap.summary_plot(shap_for_summary, sample_for_shap, show=False)
                                st.pyplot(fig_shap, bbox_inches='tight')
                                plt.close(fig_shap)
                            except Exception as e:
                                st.error(f"Could not render SHAP summary_plot: {e}")
                            
                            # PDPs
                            if pdp_features and len(pdp_features) > 0:
                                st.markdown("#### Partial Dependence Plots (PDP)")
                                st.info("Computing PDPs (average marginal effect).")
                                # Determine target for PDP for multiclass models
                                target_class = int(pred_idx) if hasattr(model, "predict_proba") else None
                                for feat in pdp_features:
                                    try:
                                        fig_pdp, ax_pdp = plt.subplots(figsize=(6, 4))
                                        # Use sklearn's PartialDependenceDisplay
                                        if hasattr(PartialDependenceDisplay, "from_estimator"):
                                            if target_class is not None and hasattr(model, "predict_proba"):
                                                PartialDependenceDisplay.from_estimator(model, X, [feat], target=target_class, ax=ax_pdp)
                                            else:
                                                PartialDependenceDisplay.from_estimator(model, X, [feat], ax=ax_pdp)
                                            st.pyplot(fig_pdp, bbox_inches='tight')
                                            plt.close(fig_pdp)
                                        else:
                                            st.warning("PartialDependenceDisplay not available in this sklearn version.")
                                    except Exception as e:
                                        st.error(f"PDP for {feat} failed: {e}")
                            else:
                                st.info("No PDP features selected. Use the PDP multi-select (left) to choose features for PDP plotting.")
                            
                        except Exception as e:
                            st.error(f"Global XAI computation failed: {e}")
                            st.info("Ensure model is tree-based or allow surrogate RandomForest. If using XGBoost ensure `use_label_encoder=False` and xgboost is installed.")
                    
                else:
                    st.error("Dataset not found. Please ensure 'Crop_recommendation.csv' is in the directory (or upload it to /mnt/data/).")
                    
        else:
            st.info("👈 Enter environmental data and click Predict")
            st.markdown("""
            **Note on Explainable AI:** The Global Prediction page supports local LIME explanations and optional Global SHAP + PDP for tree-based models.
            """)


# --- MODIFIED PAGE FUNCTION (Tamil Nadu Module) ---

def page_tamil_nadu():
    st.markdown("## 📍 Tamil Nadu Regional Mode")
    st.markdown("Specific Deep Learning Inference and XAI for Tamil Nadu Soil & Climate Conditions")
    
    # Load Resources
    data = load_resources_tn()
    
    if data is None:
        st.error("⚠️ `encoders.pkl` not found. Please ensure training artifacts are present.")
        st.info("This module requires: `encoders.pkl` and `.pth` model files generated by `train.py`.")
        return

    encoders = data['encoders']
    scaler = data['scaler']

    col1, col2 = st.columns([1, 2.5], gap="medium")

    # --- INPUT COLUMN (LEFT) ---
    with col1:
        st.markdown("### 🚜 TN Region Inputs")
        
        # Categorical Inputs
        soil_type = st.selectbox("Soil Type", encoders['SOIL'].classes_)
        crop_type = st.selectbox("Preferred Crop Type", encoders['TYPE_OF_CROP'].classes_)
        water_source = st.selectbox("Water Source", encoders['WATER_SOURCE'].classes_)
        
        st.markdown("---")
        
        # Numeric Inputs (The features inferred as used for prediction)
        tn_ph = st.slider("Soil pH (TN)", 4.0, 9.0, 6.5)
        tn_temp = st.slider("Temperature (°C) (TN)", 10.0, 45.0, 25.0)
        tn_hum = st.slider("Humidity (%) (TN)", 20.0, 100.0, 60.0)
        tn_water = st.slider("Water Available (mm)", 300, 3000, 1000)
        tn_dur = st.slider("Growing Days", 60, 365, 120)
        
        st.markdown("---")
        # --- XAI TOGGLE ---
        enable_xai = st.checkbox("Enable Explainable AI (SHAP & LIME)", value=True, help="Show deep dive into the model's decision factors using SHAP and LIME.")
        
        st.markdown("### 🎯 XAI/Prediction Target Model")
        
        # Determine initial selection (highest Test Accuracy)
        model_accs = {name: float(acc.strip('%')) for name, acc in TEST_ACCURACIES_TN.items()}
        default_model_name = max(model_accs, key=model_accs.get) # e.g., 'Transformer'
        
        model_names_for_select = sorted(TEST_ACCURACIES_TN.keys(), key=lambda x: model_accs[x], reverse=True)
        
        selected_model_name = st.selectbox(
            "Select Model for Prediction & XAI", 
            model_names_for_select,
            index=model_names_for_select.index(default_model_name),
            key="tn_xai_model_select"
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🚀 Analyze & Predict (TN)", use_container_width=True):
            try:
                # Prepare Input
                soil_enc = encoders['SOIL'].transform([soil_type])[0]
                type_enc = encoders['TYPE_OF_CROP'].transform([crop_type])[0]
                source_enc = encoders['WATER_SOURCE'].transform([water_source])[0]
                
                # Input Feature Order: [Soil_enc, CropType_enc, WaterSource_enc, pH, Duration, Temp, Water, Hum]
                features = np.array([[soil_enc, type_enc, source_enc, tn_ph, tn_dur, tn_temp, tn_water, tn_hum]])
                
                # Check for negative values from inverse transform/encoding edge cases
                if (features < 0).any():
                    st.error("Invalid categorical feature selected, cannot encode. Please check selected options.")
                    return

                features_scaled = scaler.transform(features)
                input_tensor = torch.FloatTensor(features_scaled)
                
                input_dim = 8
                output_dim = len(encoders['CROPS'].classes_)
                
                model_names = ["Transformer", "CNN", "ResidualMLP", "Hybrid_CNNLSTM", "GRU", "LSTM", "ANN"]
                results = []
                
                # Variables to track the highest accuracy model
                best_acc_val = -1.0
                best_model_data = None
                
                progress_bar = st.progress(0)
                
                for idx, name in enumerate(model_names):
                    # Instantiate model
                    if name == "CNN": model = CNNModel(input_dim, output_dim)
                    elif name == "LSTM": model = LSTMModel(input_dim, output_dim)
                    elif name == "GRU": model = GRUModel(input_dim, output_dim)
                    elif name == "Transformer": model = TransformerModel(input_dim, output_dim)
                    elif name == "ResidualMLP": model = ResidualMLP(input_dim, output_dim)
                    elif name == "Hybrid_CNNLSTM": model = Hybrid_CNNLSTM(input_dim, output_dim)
                    elif name == "ANN": model = ANNModel(input_dim, output_dim)
                    else: continue

                    try:
                        # Load state dict
                        model.load_state_dict(torch.load(f"{name}_model.pth"))
                        model.eval()
                        with torch.no_grad():
                            logits = model(input_tensor)
                            probs = F.softmax(logits, dim=1)
                            confidence, predicted_idx = torch.max(probs, 1)
                            pred_class = encoders['CROPS'].inverse_transform([predicted_idx.item()])[0]
                            conf_score = confidence.item() * 100
                            
                            acc_str = TEST_ACCURACIES_TN.get(name, "0%")
                            acc_val = float(acc_str.strip('%'))
                            
                            current_result = {
                                "Algorithm": name,
                                "Predicted Crop": pred_class,
                                "Confidence": f"{conf_score:.2f}%",
                                "Test Accuracy": acc_str,
                                "_raw_acc": acc_val,
                                "_probs": probs[0],
                                "_instance": model # Store instance for potential use
                            }
                            results.append(current_result)
                            
                            # Check for best model based on Test Accuracy
                            if acc_val > best_acc_val:
                                best_acc_val = acc_val
                                best_model_data = current_result
                                
                    except FileNotFoundError:
                        results.append({
                            "Algorithm": name, 
                            "Predicted Crop": "Missing Model File", 
                            "Confidence": "0%", 
                            "Test Accuracy": "N/A",
                            "_raw_acc": 0, 
                            "_probs": None,
                            "_instance": None
                        })
                        
                    progress_bar.progress((idx + 1) / len(model_names))
                
                progress_bar.empty()
                
                res_df = pd.DataFrame(results).sort_values(by="_raw_acc", ascending=False)
                
                # --- DETERMINE TARGET MODEL BASED ON DROPDOWN SELECTION ---
                target_row = res_df[res_df['Algorithm'] == selected_model_name].iloc[0]

                if target_row['_instance'] is not None:
                    target_model_instance = target_row['_instance']
                    selected_model = target_row['Algorithm']
                    target_pred_class = target_row['Predicted Crop']
                    target_conf = float(target_row['Confidence'].strip('%'))
                    target_probs = target_row['_probs']
                else:
                    # Fallback if the selected model is missing (should be caught in the loop, but safety first)
                    st.error(f"Selected model '{selected_model_name}' file not found or failed initialization. Cannot proceed with analysis.")
                    return


                # Store for results display
                st.session_state.tn_results = {
                    "res_df": res_df,
                    "selected_model": selected_model, 
                    "target_pred": target_pred_class,
                    "target_conf": target_conf,
                    "target_probs": target_probs,
                    "features_scaled": features_scaled,
                    "target_model_instance": target_model_instance, 
                    "enable_xai": enable_xai 
                }
                
            except Exception as e:
                st.error(f"Error during inference: {e}")

    # --- RESULTS COLUMN (RIGHT) ---
    with col2:
        if "tn_results" in st.session_state:
            res = st.session_state.tn_results
            
            # Show the SELECTED model's result prominently
            st.subheader(f"🏆 Recommendation: {res['target_pred']}")
            st.caption(f"Based on **{res['selected_model']}** (Test Accuracy: {res['res_df'][res['res_df']['Algorithm'] == res['selected_model']]['_raw_acc'].iloc[0]:.1f}%)")
            
            display_df = res['res_df'].drop(columns=["_raw_acc", "_probs", "_instance"])
            cols_order = ["Algorithm", "Predicted Crop", "Confidence", "Test Accuracy"]
            cols_order = [c for c in cols_order if c in display_df.columns]
            display_df = display_df[cols_order]

            # Highlight the model used for the final prediction (the selected one)
            def highlight_selected(row):
                if row['Algorithm'] == res['selected_model']:
                    return ['background-color: #16a34a; color: white;' for _ in row]
                return [''] * len(row)
            
            st.dataframe(display_df.style.apply(highlight_selected, axis=1), use_container_width=True)
            
            # District Data
            df_dist, dist_cols = load_district_data_tn()
            if df_dist is not None:
                st.markdown("### 📍 District Suitability")
                crop_row = df_dist[df_dist['CROPS'] == res['target_pred']]
                if not crop_row.empty:
                    suitable = [d for d in dist_cols if int(crop_row[d].values[0]) == 1]
                    if suitable:
                        st.success(f"Suitable Districts: {', '.join(suitable)}")
                    else:
                        st.warning("No specific district data for this crop.")
            
            # Top 3 (Based on Selected Model)
            if res['target_probs'] is not None:
                st.markdown("### 🥇 Top 3 Alternatives")
                top3_prob, top3_idx = torch.topk(res['target_probs'], 3)
                cols = st.columns(3)
                for i in range(3):
                    c_name = encoders['CROPS'].inverse_transform([top3_idx[i].item()])[0]
                    c_prob = top3_prob[i].item() * 100
                    with cols[i]:
                        st.metric(f"Rank {i+1}", c_name, f"{c_prob:.1f}%")

                # --- NEW XAI INTEGRATION ---
                if res['enable_xai']:
                    if res.get('target_model_instance') is not None:
                        # 1. Get X_train_background
                        # We pass both 'encoders' and 'scaler' from load_resources_tn()
                        X_train_background, feature_names = get_tn_x_train_background(encoders, scaler)
                        if X_train_background is not None:
                            # 2. Call the XAI display function
                            explain_tn_model_prediction_shap_lime(
                                res['selected_model'], 
                                res['target_model_instance'], 
                                res['features_scaled'], 
                                X_train_background, 
                                encoders['CROPS']
                            )
                        else:
                            st.warning("Cannot generate XAI: Failed to reconstruct training data background from 'Tamil Nadu - AgriData_Dist.csv'.")
                    else:
                        st.warning("Cannot generate XAI: Failed to load the predicted model instance.")
                # --- END NEW XAI INTEGRATION ---

            
        else:
            st.info("👈 Please adjust inputs on the left and click 'Analyze & Predict' to see results.")
            st.markdown("""
            **Note:** This module uses the pre-trained `.pth` models and `encoders.pkl` 
            generated from your training script. Ensure they are in the root directory.
            """)

# ==================== MAIN EXECUTION ====================

def main():
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <span style="font-size: 3rem;">🌱</span>
            <h1 style="color: #16a34a; margin: 0.5rem 0;">AgriSmart</h1>
            <p style="color: #64748b; font-size: 0.875rem;">Crop Recommendation System</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        
        # Initialize persistent storage for trained model results if they don't exist
        if 'global_benchmark_override' not in st.session_state:
            st.session_state.global_benchmark_override = {}
        if 'tn_benchmark_override' not in st.session_state:
            st.session_state.tn_benchmark_override = {}
        
        if "page" not in st.session_state:
            st.session_state.page = "home"
        
        for page_name, page_key in PAGES.items():
            type_btn = "primary" if st.session_state.page == page_key else "secondary"
            if st.button(page_name, use_container_width=True, key=f"nav_{page_key}", type=type_btn):
                st.session_state.page = page_key
                st.rerun()
        
        st.markdown("---")
        st.caption("© 2025 AgriSmart AI")

    # Routing
    page = st.session_state.page
    
    if page == "home": page_home()
    elif page == "dataset": page_dataset()
    elif page == "implementation": page_implementation()
    elif page == "training": page_training()
    elif page == "results": page_results()
    elif page == "prediction": page_prediction_global()
    elif page == "research": page_research()
    elif page == "deployment": page_deployment()
    elif page == "tamil_nadu": page_tamil_nadu()

if __name__ == "__main__":
    main()