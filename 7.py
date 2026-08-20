import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="AI Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ AI Disease Prediction System")
st.write("Upload your dataset and train multiple ML models.")

# =========================================================
# UPLOAD DATASET
# =========================================================

st.header("📂 Upload Dataset")

uploaded_file = st.file_uploader(
    "Choose your CSV dataset",
    type=["csv"]
)

if uploaded_file is None:
    st.info("👆 Please upload a CSV file.")
    st.stop()

# =========================================================
# LOAD DATA
# =========================================================

try:
    data = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"❌ Error loading dataset: {e}")
    st.stop()

st.success("✅ Dataset uploaded successfully!")

# =========================================================
# DATASET OVERVIEW
# =========================================================

st.header("📊 Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Rows", data.shape[0])
c2.metric("Columns", data.shape[1])
c3.metric(
    "Missing Values",
    int(data.isnull().sum().sum())
)
c4.metric(
    "Duplicate Rows",
    int(data.duplicated().sum())
)

# =========================================================
# SHOW COMPLETE DATASET
# =========================================================

with st.expander("📋 View Complete Dataset"):

    st.dataframe(
        data,
        use_container_width=True
    )

# =========================================================
# DATA TYPES
# =========================================================

with st.expander("🔎 Column Information"):

    info_df = pd.DataFrame({
        "Column": data.columns,
        "Data Type": data.dtypes.astype(str),
        "Missing": data.isnull().sum().values,
        "Unique Values": [
            data[col].nunique()
            for col in data.columns
        ]
    })

    st.dataframe(
        info_df,
        use_container_width=True
    )

# =========================================================
# TARGET SELECTION
# =========================================================

st.header("🎯 Select Target Column")

target_column = st.selectbox(
    "Which column should the model predict?",
    data.columns
)

st.info(
    f"🎯 Selected target: **{target_column}**"
)

# =========================================================
# PREPARE DATA
# =========================================================

X = data.drop(
    target_column,
    axis=1
)

y = data[target_column]

# =========================================================
# HANDLE CATEGORICAL FEATURES
# =========================================================

categorical_columns = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

if categorical_columns:

    st.info(
        "🔄 Categorical columns detected. "
        "They will be automatically encoded."
    )

    X = pd.get_dummies(
        X,
        columns=categorical_columns,
        drop_first=True
    )

# =========================================================
# HANDLE MISSING VALUES
# =========================================================

for column in X.columns:

    if X[column].isnull().any():

        if pd.api.types.is_numeric_dtype(
            X[column]
        ):

            X[column] = X[column].fillna(
                X[column].median()
            )

        else:

            X[column] = X[column].fillna(
                X[column].mode()[0]
            )

# Target missing values

if y.isnull().any():

    st.warning(
        "⚠️ Rows with missing target values "
        "were removed."
    )

    valid_rows = y.notnull()

    X = X[valid_rows]
    y = y[valid_rows]

# =========================================================
# ENCODE TARGET IF NEEDED
# =========================================================

target_mapping = None

if not pd.api.types.is_numeric_dtype(y):

    target_mapping = {
        value: index
        for index, value in enumerate(
            y.unique()
        )
    }

    y = y.map(target_mapping)

# =========================================================
# CHECK CLASSIFICATION
# =========================================================

if y.nunique() < 2:

    st.error(
        "❌ Target column must contain at least "
        "two different classes."
    )

    st.stop()

if y.nunique() > 10:

    st.error(
        "❌ This app is designed for classification "
        "datasets with up to 10 classes."
    )

    st.stop()

# =========================================================
# SETTINGS
# =========================================================

st.header("⚙️ Training Settings")

col1, col2 = st.columns(2)

with col1:

    test_size = st.slider(
        "Testing Data (%)",
        10,
        40,
        20
    )

with col2:

    random_state = st.number_input(
        "Random State",
        1,
        100,
        42
    )

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

try:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size / 100,
        random_state=random_state,
        stratify=y
    )

except:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size / 100,
        random_state=random_state
    )

# =========================================================
# STANDARDIZATION
# =========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

# =========================================================
# MODELS
# =========================================================

models = {

    "MLP Neural Network":
        MLPClassifier(
            hidden_layer_sizes=(50, 25),
            max_iter=1000,
            random_state=random_state
        ),

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000
        ),

    "KNN":
        KNeighborsClassifier(
            n_neighbors=5
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=random_state
        )
}

# =========================================================
# TRAIN
# =========================================================

st.header("🧠 Train Models")

if st.button(
    "🚀 Train All Models",
    use_container_width=True
):

    results = []
    trained_models = {}
    predictions = {}

    progress = st.progress(0)

    for i, (name, model) in enumerate(
        models.items()
    ):

        if name == "Random Forest":

            model.fit(
                X_train,
                y_train
            )

            prediction = model.predict(
                X_test
            )

        else:

            model.fit(
                X_train_scaled,
                y_train
            )

            prediction = model.predict(
                X_test_scaled
            )

        accuracy = accuracy_score(
            y_test,
            prediction
        )

        precision = precision_score(
            y_test,
            prediction,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            prediction,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            prediction,
            average="weighted",
            zero_division=0
        )

        results.append({
            "Model": name,
            "Accuracy (%)": round(
                accuracy * 100, 2
            ),
            "Precision (%)": round(
                precision * 100, 2
            ),
            "Recall (%)": round(
                recall * 100, 2
            ),
            "F1 Score (%)": round(
                f1 * 100, 2
            )
        })

        trained_models[name] = model
        predictions[name] = prediction

        progress.progress(
            (i + 1) / len(models)
        )

    # =====================================================
    # RESULTS
    # =====================================================

    results_df = pd.DataFrame(
        results
    )

    st.session_state.results = results_df
    st.session_state.models = trained_models
    st.session_state.predictions = predictions
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test
    st.session_state.scaler = scaler

    st.success(
        "🎉 All models trained successfully!"
    )

# =========================================================
# DISPLAY RESULTS
# =========================================================

if "results" in st.session_state:

    results_df = st.session_state.results

    st.header("📊 Complete Model Results")

    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True
    )

    # =====================================================
    # BEST MODEL
    # =====================================================

    best_index = results_df[
        "Accuracy (%)"
    ].idxmax()

    best_model = results_df.loc[
        best_index,
        "Model"
    ]

    best_accuracy = results_df.loc[
        best_index,
        "Accuracy (%)"
    ]

    st.success(
        f"🏆 Best Model: {best_model} "
        f"— Accuracy: {best_accuracy}%"
    )

    # =====================================================
    # GRAPH
    # =====================================================

    st.subheader(
        "📈 Model Performance Comparison"
    )

    fig, ax = plt.subplots()

    ax.bar(
        results_df["Model"],
        results_df["Accuracy (%)"]
    )

    ax.set_ylabel(
        "Accuracy (%)"
    )

    ax.set_title(
        "Model Accuracy Comparison"
    )

    plt.xticks(
        rotation=20
    )

    st.pyplot(fig)

    # =====================================================
    # MODEL SELECTION
    # =====================================================

    st.header(
        "🔍 Detailed Model Analysis"
    )

    selected_model_name = st.selectbox(
        "Select model",
        list(
            st.session_state.models.keys()
        )
    )

    selected_model = (
        st.session_state.models[
            selected_model_name
        ]
    )

    selected_prediction = (
        st.session_state.predictions[
            selected_model_name
        ]
    )

    y_test_saved = (
        st.session_state.y_test
    )

    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    st.subheader(
        "🔲 Confusion Matrix"
    )

    cm = confusion_matrix(
        y_test_saved,
        selected_prediction
    )

    fig, ax = plt.subplots()

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    disp.plot(
        ax=ax
    )

    ax.set_title(
        selected_model_name
    )

    st.pyplot(fig)

    # =====================================================
    # CLASSIFICATION REPORT
    # =====================================================

    st.subheader(
        "📋 Classification Report"
    )

    report = classification_report(
        y_test_saved,
        selected_prediction,
        output_dict=True,
        zero_division=0
    )

    st.dataframe(
        pd.DataFrame(report).transpose(),
        use_container_width=True
    )

    # =====================================================
    # ACTUAL VS PREDICTED
    # =====================================================

    st.subheader(
        "🔍 Actual vs Predicted"
    )

    comparison = pd.DataFrame({
        "Actual": y_test_saved.values,
        "Predicted": selected_prediction
    })

    st.dataframe(
        comparison,
        use_container_width=True
    )

    # =====================================================
    # DOWNLOAD
    # =====================================================

    csv = comparison.to_csv(
        index=False
    )

    st.download_button(
        "📥 Download Prediction Results",
        csv,
        "prediction_results.csv",
        "text/csv"
    )