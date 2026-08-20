import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.neural_network import MLPClassifier

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
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 AI-Powered Disease Prediction System")

st.write(
    "Upload different types of disease datasets and train "
    "multiple AI models automatically."
)

st.warning(
    "⚠️ Educational/research application only. "
    "It is not a medical diagnosis."
)

# =========================================================
# UPLOAD DATASET
# =========================================================

st.header("📂 Upload Disease Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel dataset",
    type=["csv", "xlsx"]
)

if uploaded_file is None:
    st.info("👆 Please upload your disease dataset.")
    st.stop()

# =========================================================
# LOAD DATA
# =========================================================

try:

    if uploaded_file.name.lower().endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

except Exception as e:

    st.error(f"❌ Error loading dataset: {e}")
    st.stop()

st.success(
    f"✅ Dataset uploaded successfully: {uploaded_file.name}"
)

# =========================================================
# BASIC CLEANING
# =========================================================

# Remove completely empty columns
data = data.dropna(
    axis=1,
    how="all"
)

# Remove completely empty rows
data = data.dropna(
    axis=0,
    how="all"
)

# Remove duplicate rows
duplicate_count = data.duplicated().sum()

if duplicate_count > 0:

    data = data.drop_duplicates()

    st.info(
        f"🧹 Removed {duplicate_count} duplicate rows."
    )

# =========================================================
# DATASET OVERVIEW
# =========================================================

st.header("📊 Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Rows",
    data.shape[0]
)

c2.metric(
    "Columns",
    data.shape[1]
)

c3.metric(
    "Missing Values",
    int(data.isnull().sum().sum())
)

c4.metric(
    "Duplicate Rows",
    int(data.duplicated().sum())
)

# =========================================================
# DATA PREVIEW
# =========================================================

with st.expander("👀 View Complete Dataset"):

    st.dataframe(
        data,
        use_container_width=True
    )

# =========================================================
# COLUMN INFORMATION
# =========================================================

with st.expander("🔎 Column Information"):

    info_df = pd.DataFrame({

        "Column":
            data.columns,

        "Data Type":
            data.dtypes.astype(str),

        "Missing":
            data.isnull().sum().values,

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

st.header("🎯 Disease Target")

target_column = st.selectbox(
    "Select the column containing the disease/class",
    data.columns
)

st.success(
    f"🩺 Target selected: {target_column}"
)

# =========================================================
# PREPARE X AND Y
# =========================================================

X = data.drop(
    columns=[target_column]
).copy()

y = data[target_column].copy()

# Remove rows with missing target
valid_rows = y.notna()

X = X.loc[valid_rows].reset_index(drop=True)

y = y.loc[valid_rows].reset_index(drop=True)

# =========================================================
# TARGET ENCODING
# =========================================================

target_encoder = LabelEncoder()

y = target_encoder.fit_transform(
    y.astype(str)
)

class_names = target_encoder.classes_

number_of_classes = len(
    class_names
)

if number_of_classes < 2:

    st.error(
        "❌ The target must contain at least two disease classes."
    )

    st.stop()

st.info(
    f"🩺 Detected {number_of_classes} disease classes."
)

# =========================================================
# IDENTIFY FEATURE TYPES
# =========================================================

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

st.header("🔍 Automatic Feature Detection")

col1, col2 = st.columns(2)

with col1:

    st.write("🔢 **Numerical Features**")

    if numeric_features:
        st.write(numeric_features)
    else:
        st.write("None")

with col2:

    st.write("🏷️ **Categorical Features**")

    if categorical_features:
        st.write(categorical_features)
    else:
        st.write("None")

# =========================================================
# PREPROCESSING
# =========================================================

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="most_frequent"
        )
    ),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore"
        )
    )
])

preprocessor = ColumnTransformer(
    transformers=[

        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),

        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)

# =========================================================
# TRAIN TEST SPLIT
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
        1000,
        42
    )

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
# MODELS
# =========================================================

base_models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=2000,
            random_state=random_state
        ),

    "KNN":
        KNeighborsClassifier(
            n_neighbors=5
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=random_state
        ),

    "Extra Trees":
        ExtraTreesClassifier(
            n_estimators=200,
            random_state=random_state
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(
            random_state=random_state
        ),

    "Neural Network":
        MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=1500,
            random_state=random_state
        )
}

# =========================================================
# VOTING ENSEMBLE
# =========================================================

voting_estimator = VotingClassifier(

    estimators=[

        (
            "lr",
            LogisticRegression(
                max_iter=2000
            )
        ),

        (
            "rf",
            RandomForestClassifier(
                n_estimators=200,
                random_state=random_state
            )
        ),

        (
            "extra",
            ExtraTreesClassifier(
                n_estimators=200,
                random_state=random_state
            )
        )
    ],

    voting="soft"
)

base_models[
    "Voting Ensemble"
] = voting_estimator

# =========================================================
# TRAIN MODELS
# =========================================================

st.header("🧠 AI Model Training")

if st.button(
    "🚀 Train All Disease Models",
    use_container_width=True
):

    results = []

    trained_models = {}

    predictions = {}

    probabilities = {}

    progress = st.progress(0)

    total = len(base_models)

    for i, (name, estimator) in enumerate(
        base_models.items()
    ):

        try:

            # Every model gets the same automatic
            # preprocessing pipeline.

            model = Pipeline([

                (
                    "preprocessor",
                    preprocessor
                ),

                (
                    "model",
                    estimator
                )
            ])

            model.fit(
                X_train,
                y_train
            )

            prediction = model.predict(
                X_test
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

                "Model":
                    name,

                "Accuracy (%)":
                    round(
                        accuracy * 100,
                        2
                    ),

                "Precision (%)":
                    round(
                        precision * 100,
                        2
                    ),

                "Recall (%)":
                    round(
                        recall * 100,
                        2
                    ),

                "F1 Score (%)":
                    round(
                        f1 * 100,
                        2
                    )
            })

            trained_models[name] = model

            predictions[name] = prediction

            if hasattr(
                model,
                "predict_proba"
            ):

                probabilities[name] = (
                    model.predict_proba(
                        X_test
                    )
                )

        except Exception as e:

            st.warning(
                f"⚠️ {name} failed: {e}"
            )

        progress.progress(
            (i + 1) / total
        )

    # =====================================================
    # SAVE RESULTS
    # =====================================================

    results_df = pd.DataFrame(
        results
    )

    st.session_state.results = (
        results_df
    )

    st.session_state.models = (
        trained_models
    )

    st.session_state.predictions = (
        predictions
    )

    st.session_state.probabilities = (
        probabilities
    )

    st.session_state.X_test = X_test

    st.session_state.y_test = y_test

    st.session_state.feature_names = (
        X.columns.tolist()
    )

    st.success(
        "🎉 All models trained successfully!"
    )

# =========================================================
# MODEL RESULTS
# =========================================================

if "results" in st.session_state:

    results_df = (
        st.session_state.results
    )

    st.header("🏆 Model Comparison")

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
        f"🏆 Best Model: **{best_model}** "
        f"— Accuracy: **{best_accuracy}%**"
    )

    # =====================================================
    # PERFORMANCE GRAPH
    # =====================================================

    st.subheader(
        "📈 Model Accuracy Comparison"
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
        "Disease Prediction Model Performance"
    )

    plt.xticks(
        rotation=30
    )

    st.pyplot(fig)

    # =====================================================
    # DETAILED MODEL
    # =====================================================

    st.header(
        "🔍 Detailed Model Analysis"
    )

    selected_model_name = st.selectbox(
        "Select a model",
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

    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    st.subheader(
        "🔲 Confusion Matrix"
    )

    cm = confusion_matrix(
        st.session_state.y_test,
        selected_prediction
    )

    fig, ax = plt.subplots()

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot(
        ax=ax,
        xticks_rotation=45
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

        st.session_state.y_test,

        selected_prediction,

        target_names=class_names,

        output_dict=True,

        zero_division=0
    )

    st.dataframe(
        pd.DataFrame(
            report
        ).transpose(),

        use_container_width=True
    )

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    st.header(
        "🔥 Feature Importance"
    )

    actual_estimator = (
        selected_model.named_steps["model"]
    )

    if hasattr(
        actual_estimator,
        "feature_importances_"
    ):

        importance = (
            actual_estimator.feature_importances_
        )

        try:

            transformed_names = (
                selected_model
                .named_steps[
                    "preprocessor"
                ]
                .get_feature_names_out()
            )

            importance_df = pd.DataFrame({

                "Feature":
                    transformed_names,

                "Importance":
                    importance
            })

            importance_df = (
                importance_df
                .sort_values(
                    "Importance",
                    ascending=False
                )
                .head(20)
            )

            st.dataframe(
                importance_df,
                use_container_width=True
            )

            fig, ax = plt.subplots()

            ax.barh(
                importance_df["Feature"],
                importance_df["Importance"]
            )

            ax.set_xlabel(
                "Importance"
            )

            ax.set_title(
                "Top Disease Prediction Features"
            )

            ax.invert_yaxis()

            st.pyplot(fig)

        except Exception:

            st.info(
                "Feature names could not be displayed."
            )

    else:

        st.info(
            "Feature importance is not directly available "
            "for this model."
        )

    # =====================================================
    # ACTUAL VS PREDICTED
    # =====================================================

    st.subheader(
        "🔍 Actual vs Predicted Disease"
    )

    actual_labels = (
        class_names[
            st.session_state.y_test
        ]
    )

    predicted_labels = (
        class_names[
            selected_prediction
        ]
    )

    comparison = pd.DataFrame({

        "Actual Disease":
            actual_labels,

        "Predicted Disease":
            predicted_labels
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
        "disease_predictions.csv",
        "text/csv"
    )

# =========================================================
# PATIENT PREDICTION
# =========================================================

if "models" in st.session_state:

    st.header(
        "👤 Patient Disease Prediction"
    )

    st.write(
        "Enter the patient's information. "
        "The app automatically creates number fields "
        "or dropdowns depending on the original dataset."
    )

    prediction_model_name = st.selectbox(
        "Choose prediction model",
        list(
            st.session_state.models.keys()
        ),
        key="patient_model"
    )

    prediction_model = (
        st.session_state.models[
            prediction_model_name
        ]
    )

    patient_values = {}

    st.subheader(
        "📝 Patient Information"
    )

    input_columns = st.columns(2)

    for i, column in enumerate(
        X.columns
    ):

        with input_columns[i % 2]:

            # ---------------------------------------------
            # NUMERIC FEATURE
            # ---------------------------------------------

            if pd.api.types.is_numeric_dtype(
                X[column]
            ):

                min_value = float(
                    X[column].min()
                )

                max_value = float(
                    X[column].max()
                )

                median_value = float(
                    X[column].median()
                )

                patient_values[column] = (
                    st.number_input(

                        column,

                        min_value=min_value,

                        max_value=max_value,

                        value=median_value
                    )
                )

            # ---------------------------------------------
            # CATEGORICAL FEATURE
            # ---------------------------------------------

            else:

                values = (
                    X[column]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                values = sorted(
                    values
                )

                if len(values) > 100:

                    patient_values[column] = (
                        st.text_input(
                            column
                        )
                    )

                else:

                    patient_values[column] = (
                        st.selectbox(
                            column,
                            values
                        )
                    )

    # =====================================================
    # PREDICT
    # =====================================================

    if st.button(
        "🔮 Predict Disease",
        use_container_width=True
    ):

        patient_df = pd.DataFrame(
            [patient_values]
        )

        try:

            prediction = (
                prediction_model.predict(
                    patient_df
                )
            )

            predicted_class = int(
                prediction[0]
            )

            disease = (
                class_names[
                    predicted_class
                ]
            )

            st.success(
                f"🩺 Predicted Disease: **{disease}**"
            )

            # ---------------------------------------------
            # PROBABILITY
            # ---------------------------------------------

            if hasattr(
                prediction_model,
                "predict_proba"
            ):

                probability = (
                    prediction_model
                    .predict_proba(
                        patient_df
                    )[0]
                )

                probability_df = pd.DataFrame({

                    "Disease":
                        class_names,

                    "Probability (%)":
                        np.round(
                            probability * 100,
                            2
                        )
                })

                probability_df = (
                    probability_df
                    .sort_values(
                        "Probability (%)",
                        ascending=False
                    )
                )

                st.subheader(
                    "📊 Disease Probability"
                )

                st.dataframe(
                    probability_df,
                    use_container_width=True,
                    hide_index=True
                )

                # Top probability

                top_probability = (
                    probability_df.iloc[0]
                )

                st.info(
                    f"Highest predicted probability: "
                    f"**{top_probability['Disease']} "
                    f"({top_probability['Probability (%)']}%)**"
                )

        except Exception as e:

            st.error(
                f"❌ Prediction error: {e}"
            )

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🩺 AI Disease Prediction System | "
    "Educational and research use only"
)