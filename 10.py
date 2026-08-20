import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    LabelEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

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
    classification_report
)

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Disease Predictor",
    page_icon="🩺",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}

.card {
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #ddd;
    margin-bottom: 15px;
}

.big-result {
    font-size: 30px;
    font-weight: 800;
    text-align: center;
    padding: 25px;
    border-radius: 15px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">🩺 AI Disease Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload a disease dataset • Train multiple AI models • Predict disease'
    '</div>',
    unsafe_allow_html=True
)

st.warning(
    "⚠️ Educational and research use only. "
    "This application is not a medical diagnosis."
)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🩺 AI Disease Predictor")

st.sidebar.markdown("""
### Features

📂 Dataset Upload  
🔍 Automatic Data Analysis  
🧹 Smart Preprocessing  
🤖 Multiple ML Models  
🏆 Best Model Detection  
📊 Performance Analysis  
🔥 Feature Importance  
👤 Patient Prediction  
🔮 Probability Prediction  
📥 Download Results
""")

# =========================================================
# UPLOAD
# =========================================================

st.header("📂 Step 1 — Upload Disease Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"]
)

if uploaded_file is None:

    st.info(
        "👆 Upload a disease dataset to start."
    )

    st.stop()

# =========================================================
# LOAD DATA
# =========================================================

try:

    if uploaded_file.name.lower().endswith(".csv"):

        df = pd.read_csv(
            uploaded_file
        )

    else:

        df = pd.read_excel(
            uploaded_file
        )

except Exception as e:

    st.error(
        f"Unable to read dataset: {e}"
    )

    st.stop()

# =========================================================
# CLEAN COLUMN NAMES
# =========================================================

df.columns = [
    str(col).strip()
    for col in df.columns
]

# Remove empty rows/columns

df = df.dropna(
    axis=0,
    how="all"
)

df = df.dropna(
    axis=1,
    how="all"
)

# Remove duplicates

duplicate_count = int(
    df.duplicated().sum()
)

if duplicate_count > 0:

    df = df.drop_duplicates()

# =========================================================
# SUCCESS
# =========================================================

st.success(
    f"✅ Dataset loaded successfully — "
    f"{df.shape[0]} rows × {df.shape[1]} columns"
)

# =========================================================
# DATASET METRICS
# =========================================================

st.header("📊 Step 2 — Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Rows",
    df.shape[0]
)

c2.metric(
    "Columns",
    df.shape[1]
)

c3.metric(
    "Missing Values",
    int(df.isna().sum().sum())
)

c4.metric(
    "Duplicates Removed",
    duplicate_count
)

# =========================================================
# PREVIEW
# =========================================================

with st.expander(
    "👀 Preview Dataset"
):

    st.dataframe(
        df.head(100),
        use_container_width=True
    )

# =========================================================
# TARGET
# =========================================================

st.header("🎯 Step 3 — Select Disease Column")

target = st.selectbox(
    "Which column contains the disease/class?",
    df.columns
)

st.success(
    f"🎯 Prediction target: **{target}**"
)

# =========================================================
# TARGET CLEANING
# =========================================================

df = df[df[target].notna()].copy()

y_raw = df[target].astype(str)

X = df.drop(
    columns=[target]
).copy()

# =========================================================
# REMOVE ID-LIKE COLUMNS
# =========================================================

id_columns = []

for col in X.columns:

    unique_count = X[col].nunique()

    if (
        unique_count == len(X)
        and
        len(X) > 20
    ):

        id_columns.append(col)

if id_columns:

    X = X.drop(
        columns=id_columns
    )

    st.info(
        "🧹 Removed ID-like columns: "
        + ", ".join(id_columns)
    )

# =========================================================
# TARGET ENCODING
# =========================================================

target_encoder = LabelEncoder()

y = target_encoder.fit_transform(
    y_raw
)

class_names = list(
    target_encoder.classes_
)

# =========================================================
# CLASS CHECK
# =========================================================

if len(class_names) < 2:

    st.error(
        "❌ Disease column must contain at least 2 classes."
    )

    st.stop()

st.info(
    f"🩺 Detected {len(class_names)} disease classes."
)

# =========================================================
# FEATURE DETECTION
# =========================================================

numeric_columns = X.select_dtypes(
    include=np.number
).columns.tolist()

categorical_columns = X.select_dtypes(
    exclude=np.number
).columns.tolist()

st.header("🔍 Step 4 — Automatic Feature Detection")

col1, col2 = st.columns(2)

with col1:

    st.subheader(
        "🔢 Numeric"
    )

    if numeric_columns:

        for col in numeric_columns:

            st.write(
                "• " + col
            )

    else:

        st.write(
            "No numeric columns"
        )

with col2:

    st.subheader(
        "🏷️ Categorical"
    )

    if categorical_columns:

        for col in categorical_columns:

            st.write(
                "• " + col
            )

    else:

        st.write(
            "No categorical columns"
        )

# =========================================================
# PREPROCESSOR
# =========================================================

numeric_pipeline = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            StandardScaler()
        )
    ]
)

categorical_pipeline = Pipeline(
    steps=[

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
    ]
)

preprocessor = ColumnTransformer(
    transformers=[

        (
            "numeric",
            numeric_pipeline,
            numeric_columns
        ),

        (
            "categorical",
            categorical_pipeline,
            categorical_columns
        )
    ],
    remainder="drop"
)

# =========================================================
# TRAINING SETTINGS
# =========================================================

st.header("⚙️ Step 5 — Training Settings")

col1, col2 = st.columns(2)

with col1:

    test_percent = st.slider(
        "Test Dataset (%)",
        10,
        40,
        20
    )

with col2:

    random_state = st.number_input(
        "Random State",
        min_value=1,
        max_value=1000,
        value=42
    )

# =========================================================
# SAFE TRAIN TEST SPLIT
# =========================================================

class_counts = pd.Series(
    y
).value_counts()

use_stratify = (
    class_counts.min() >= 2
)

if use_stratify:

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=test_percent / 100,
            random_state=random_state,
            stratify=y
        )
    )

else:

    st.warning(
        "⚠️ Some disease classes contain only one sample. "
        "Using a normal train/test split."
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=test_percent / 100,
            random_state=random_state
        )
    )

# =========================================================
# MODELS
# =========================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=2000
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
            max_iter=1000,
            random_state=random_state
        )
}

# =========================================================
# VOTING ENSEMBLE
# =========================================================

voting = VotingClassifier(

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
                n_estimators=150,
                random_state=random_state
            )
        ),

        (
            "extra",
            ExtraTreesClassifier(
                n_estimators=150,
                random_state=random_state
            )
        )
    ],

    voting="soft"
)

models[
    "⭐ Voting Ensemble"
] = voting

# =========================================================
# TRAIN BUTTON
# =========================================================

st.header("🤖 Step 6 — Train AI Models")

train = st.button(
    "🚀 TRAIN ALL MODELS",
    use_container_width=True
)

if train:

    results = []

    trained_models = {}

    predictions = {}

    progress = st.progress(0)

    status = st.empty()

    for index, (
        model_name,
        algorithm
    ) in enumerate(
        models.items()
    ):

        status.write(
            f"Training **{model_name}**..."
        )

        try:

            pipeline = Pipeline(
                steps=[

                    (
                        "preprocessor",
                        preprocessor
                    ),

                    (
                        "model",
                        algorithm
                    )
                ]
            )

            pipeline.fit(
                X_train,
                y_train
            )

            pred = pipeline.predict(
                X_test
            )

            accuracy = accuracy_score(
                y_test,
                pred
            )

            precision = precision_score(
                y_test,
                pred,
                average="weighted",
                zero_division=0
            )

            recall = recall_score(
                y_test,
                pred,
                average="weighted",
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                pred,
                average="weighted",
                zero_division=0
            )

            results.append({

                "Model":
                    model_name,

                "Accuracy":
                    round(
                        accuracy * 100,
                        2
                    ),

                "Precision":
                    round(
                        precision * 100,
                        2
                    ),

                "Recall":
                    round(
                        recall * 100,
                        2
                    ),

                "F1 Score":
                    round(
                        f1 * 100,
                        2
                    )
            })

            trained_models[
                model_name
            ] = pipeline

            predictions[
                model_name
            ] = pred

        except Exception as error:

            st.warning(
                f"{model_name} skipped: {error}"
            )

        progress.progress(
            (index + 1) / len(models)
        )

    status.success(
        "🎉 Model training completed!"
    )

    if not results:

        st.error(
            "❌ No model could be trained. "
            "Please check your dataset."
        )

        st.stop()

    results_df = pd.DataFrame(
        results
    )

    st.session_state.results = (
        results_df
    )

    st.session_state.trained_models = (
        trained_models
    )

    st.session_state.predictions = (
        predictions
    )

    st.session_state.X_test = X_test

    st.session_state.y_test = y_test

    st.session_state.X_columns = (
        X.columns.tolist()
    )

# =========================================================
# RESULTS
# =========================================================

if "results" in st.session_state:

    results_df = (
        st.session_state.results
    )

    st.header(
        "🏆 Step 7 — AI Model Results"
    )

    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True
    )

    # =====================================================
    # BEST MODEL
    # =====================================================

    best_row = results_df.loc[
        results_df["Accuracy"].idxmax()
    ]

    best_model_name = (
        best_row["Model"]
    )

    best_accuracy = (
        best_row["Accuracy"]
    )

    st.markdown(
        f"""
        <div class="big-result">
        🏆 BEST MODEL<br>
        {best_model_name}<br>
        Accuracy: {best_accuracy}%
        </div>
        """,
        unsafe_allow_html=True
    )

    # =====================================================
    # GRAPH
    # =====================================================

    st.subheader(
        "📈 Model Performance"
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.bar(
        results_df["Model"],
        results_df["Accuracy"]
    )

    ax.set_ylabel(
        "Accuracy (%)"
    )

    ax.set_xlabel(
        "Model"
    )

    ax.set_title(
        "Disease Prediction Accuracy"
    )

    ax.set_ylim(
        0,
        100
    )

    plt.xticks(
        rotation=35,
        ha="right"
    )

    plt.tight_layout()

    st.pyplot(fig)

    # =====================================================
    # MODEL SELECTOR
    # =====================================================

    st.header(
        "🔬 Step 8 — Detailed Analysis"
    )

    selected_name = st.selectbox(
        "Select model",
        list(
            st.session_state
            .trained_models
            .keys()
        )
    )

    selected_model = (
        st.session_state
        .trained_models[
            selected_name
        ]
    )

    selected_pred = (
        st.session_state
        .predictions[
            selected_name
        ]
    )

    # =====================================================
    # CONFUSION MATRIX - SAFE
    # =====================================================

    st.subheader(
        "🔲 Confusion Matrix"
    )

    true_values = np.asarray(
        st.session_state.y_test
    )

    predicted_values = np.asarray(
        selected_pred
    )

    # ALWAYS use every possible class.
    # This prevents the label mismatch error.

    all_labels = np.arange(
        len(class_names)
    )

    cm = confusion_matrix(
        true_values,
        predicted_values,
        labels=all_labels
    )

    fig, ax = plt.subplots(
        figsize=(
            max(7, len(class_names)),
            max(6, len(class_names))
        )
    )

    image = ax.imshow(
        cm
    )

    ax.set_title(
        "Disease Confusion Matrix"
    )

    ax.set_xlabel(
        "Predicted Disease"
    )

    ax.set_ylabel(
        "Actual Disease"
    )

    ax.set_xticks(
        np.arange(len(class_names))
    )

    ax.set_yticks(
        np.arange(len(class_names))
    )

    ax.set_xticklabels(
        class_names,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(
        class_names
    )

    # Write numbers inside matrix

    for i in range(
        len(class_names)
    ):

        for j in range(
            len(class_names)
        ):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    plt.tight_layout()

    st.pyplot(fig)

    # =====================================================
    # CLASSIFICATION REPORT
    # =====================================================

    st.subheader(
        "📋 Classification Report"
    )

    report = classification_report(
        true_values,
        predicted_values,
        labels=all_labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    st.dataframe(
        report_df,
        use_container_width=True
    )

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    st.header(
        "🔥 Important Features"
    )

    algorithm = (
        selected_model
        .named_steps["model"]
    )

    if hasattr(
        algorithm,
        "feature_importances_"
    ):

        importance = (
            algorithm
            .feature_importances_
        )

        try:

            names = (
                selected_model
                .named_steps[
                    "preprocessor"
                ]
                .get_feature_names_out()
            )

            importance_df = pd.DataFrame({

                "Feature":
                    names,

                "Importance":
                    importance
            })

            importance_df = (
                importance_df
                .sort_values(
                    "Importance",
                    ascending=False
                )
                .head(15)
            )

            st.dataframe(
                importance_df,
                use_container_width=True,
                hide_index=True
            )

            fig, ax = plt.subplots(
                figsize=(9, 6)
            )

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

            plt.tight_layout()

            st.pyplot(fig)

        except Exception:

            st.info(
                "Feature importance could not be displayed "
                "for this dataset."
            )

    else:

        st.info(
            "This model does not provide direct feature importance."
        )

    # =====================================================
    # PATIENT PREDICTION
    # =====================================================

    st.header(
        "👤 Step 9 — Predict Disease for a Patient"
    )

    st.write(
        "The app automatically creates the correct input "
        "type from your dataset."
    )

    patient_data = {}

    input_cols = st.columns(2)

    for i, column in enumerate(
        X.columns
    ):

        with input_cols[
            i % 2
        ]:

            # ---------------------------------------------
            # NUMERIC
            # ---------------------------------------------

            if column in numeric_columns:

                series = pd.to_numeric(
                    X[column],
                    errors="coerce"
                )

                minimum = float(
                    series.min()
                )

                maximum = float(
                    series.max()
                )

                default = float(
                    series.median()
                )

                if minimum == maximum:

                    patient_data[
                        column
                    ] = st.number_input(
                        column,
                        value=default
                    )

                else:

                    patient_data[
                        column
                    ] = st.number_input(
                        column,
                        min_value=minimum,
                        max_value=maximum,
                        value=default
                    )

            # ---------------------------------------------
            # CATEGORICAL
            # ---------------------------------------------

            else:

                choices = (
                    X[column]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                choices = sorted(
                    choices
                )

                if len(choices) == 0:

                    patient_data[
                        column
                    ] = st.text_input(
                        column
                    )

                elif len(choices) <= 30:

                    patient_data[
                        column
                    ] = st.selectbox(
                        column,
                        choices
                    )

                else:

                    patient_data[
                        column
                    ] = st.text_input(
                        column
                    )

    # =====================================================
    # PREDICT BUTTON
    # =====================================================

    if st.button(
        "🔮 PREDICT DISEASE",
        use_container_width=True
    ):

        patient_df = pd.DataFrame(
            [patient_data]
        )

        # Ensure exact original feature order

        patient_df = patient_df[
            X.columns
        ]

        try:

            result = (
                selected_model
                .predict(
                    patient_df
                )
            )

            disease_index = int(
                result[0]
            )

            disease_name = (
                class_names[
                    disease_index
                ]
            )

            st.markdown(
                f"""
                <div class="big-result">
                🩺 PREDICTED DISEASE<br>
                {disease_name}
                </div>
                """,
                unsafe_allow_html=True
            )

            # =================================================
            # PROBABILITY
            # =================================================

            if hasattr(
                selected_model,
                "predict_proba"
            ):

                probability = (
                    selected_model
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
                    "📊 Prediction Probability"
                )

                st.dataframe(
                    probability_df,
                    use_container_width=True,
                    hide_index=True
                )

                # Probability graph

                fig, ax = plt.subplots(
                    figsize=(9, 5)
                )

                ax.bar(
                    probability_df["Disease"],
                    probability_df[
                        "Probability (%)"
                    ]
                )

                ax.set_ylabel(
                    "Probability (%)"
                )

                ax.set_xlabel(
                    "Disease"
                )

                ax.set_title(
                    "Disease Prediction Probability"
                )

                ax.set_ylim(
                    0,
                    100
                )

                plt.xticks(
                    rotation=35,
                    ha="right"
                )

                plt.tight_layout()

                st.pyplot(fig)

        except Exception as error:

            st.error(
                f"❌ Prediction failed: {error}"
            )

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🩺 AI Disease Prediction System | "
    "Machine Learning + Ensemble Intelligence | "
    "Educational Use Only"
)