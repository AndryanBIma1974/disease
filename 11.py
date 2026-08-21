import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =========================
# PAGE
# =========================

st.set_page_config(
    page_title="AI Disease Predictor",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 AI Disease Prediction System")
st.write("⚡ Fast AI disease prediction using your own dataset")

st.warning(
    "⚠️ Educational/research use only. "
    "This is not a medical diagnosis."
)

# =========================
# UPLOAD DATASET
# =========================

st.header("📂 Upload Dataset")

file = st.file_uploader(
    "Upload CSV or Excel",
    type=["csv", "xlsx"]
)

if file is None:
    st.info("Upload your disease dataset to begin.")
    st.stop()

# =========================
# LOAD DATA
# =========================

try:
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
except Exception as e:
    st.error(f"Cannot read file: {e}")
    st.stop()

# Clean
df = df.dropna(axis=0, how="all")
df = df.dropna(axis=1, how="all")
df = df.drop_duplicates()

df.columns = [str(c).strip() for c in df.columns]

# =========================
# FAST MODE
# =========================

MAX_ROWS = 20000

if len(df) > MAX_ROWS:
    df = df.sample(
        MAX_ROWS,
        random_state=42
    ).reset_index(drop=True)

    st.info(
        f"⚡ Fast mode: using {MAX_ROWS:,} rows."
    )

# =========================
# DATASET INFORMATION
# =========================

c1, c2, c3 = st.columns(3)

c1.metric("Rows", len(df))
c2.metric("Columns", len(df.columns))
c3.metric(
    "Missing Values",
    int(df.isna().sum().sum())
)

with st.expander("👀 View Dataset"):
    st.dataframe(
        df.head(100),
        use_container_width=True
    )

# =========================
# TARGET COLUMN
# =========================

st.header("🎯 Select Disease Column")

target = st.selectbox(
    "Select the column containing the disease",
    df.columns
)

# Remove rows without target
df = df[df[target].notna()].copy()

# =========================
# X / Y
# =========================

X = df.drop(
    columns=[target]
).copy()

y_text = df[target].astype(str)

# Remove ID-like columns
remove_columns = []

for col in X.columns:

    if (
        X[col].nunique() == len(X)
        and len(X) > 20
    ):
        remove_columns.append(col)

if remove_columns:

    X = X.drop(
        columns=remove_columns
    )

# =========================
# ENCODE TARGET
# =========================

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(
    y_text
)

diseases = label_encoder.classes_

if len(diseases) < 2:
    st.error(
        "The disease column needs at least 2 classes."
    )
    st.stop()

st.success(
    f"🩺 Detected {len(diseases)} disease classes"
)

# =========================
# FEATURE TYPES
# =========================

numeric = X.select_dtypes(
    include=np.number
).columns.tolist()

categorical = X.select_dtypes(
    exclude=np.number
).columns.tolist()

col1, col2 = st.columns(2)

with col1:
    st.write("🔢 **Numeric features**")
    st.write(numeric if numeric else "None")

with col2:
    st.write("🏷️ **Categorical features**")
    st.write(categorical if categorical else "None")

# =========================
# PREPROCESSING
# =========================

numeric_pipe = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])

categorical_pipe = Pipeline([
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

preprocessor = ColumnTransformer([
    (
        "num",
        numeric_pipe,
        numeric
    ),
    (
        "cat",
        categorical_pipe,
        categorical
    )
])

# =========================
# TRAIN / TEST
# =========================

test_size = st.slider(
    "Testing data (%)",
    10,
    40,
    20
)

counts = pd.Series(y).value_counts()

if counts.min() >= 2:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size / 100,
        random_state=42,
        stratify=y
    )

else:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size / 100,
        random_state=42
    )

# =========================
# FAST MODELS
# =========================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=500
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            n_jobs=-1,
            random_state=42
        ),

    "⭐ Voting Ensemble":
        VotingClassifier(

            estimators=[

                (
                    "lr",
                    LogisticRegression(
                        max_iter=500
                    )
                ),

                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=50,
                        max_depth=10,
                        n_jobs=-1,
                        random_state=42
                    )
                )
            ],

            voting="soft"
        )
}

# =========================
# TRAIN
# =========================

if st.button(
    "🚀 TRAIN AI MODELS",
    use_container_width=True
):

    results = []
    trained = {}
    predictions = {}

    progress = st.progress(0)

    for number, (
        name,
        algorithm
    ) in enumerate(models.items()):

        try:

            model = Pipeline([

                (
                    "preprocessor",
                    preprocessor
                ),

                (
                    "model",
                    algorithm
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

            results.append({
                "Model": name,
                "Accuracy (%)": round(
                    accuracy * 100,
                    2
                )
            })

            trained[name] = model
            predictions[name] = prediction

        except Exception as e:

            st.warning(
                f"{name} skipped: {e}"
            )

        progress.progress(
            (number + 1) / len(models)
        )

    if not results:

        st.error(
            "No model could be trained."
        )
        st.stop()

    st.session_state.results = pd.DataFrame(
        results
    )

    st.session_state.trained = trained
    st.session_state.predictions = predictions
    st.session_state.y_test = y_test

    st.success(
        "🎉 Training completed!"
    )

# =========================
# RESULTS
# =========================

if "results" in st.session_state:

    st.header("🏆 Model Results")

    results = st.session_state.results

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True
    )

    best = results.loc[
        results["Accuracy (%)"].idxmax()
    ]

    st.success(
        f"🏆 Best Model: **{best['Model']}** "
        f"| Accuracy: **{best['Accuracy (%)']}%**"
    )

    # Accuracy graph

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    ax.bar(
        results["Model"],
        results["Accuracy (%)"]
    )

    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("AI Model Accuracy")

    plt.xticks(
        rotation=25,
        ha="right"
    )

    plt.tight_layout()

    st.pyplot(fig)

    # =========================
    # MODEL ANALYSIS
    # =========================

    selected = st.selectbox(
        "Choose model for analysis",
        list(
            st.session_state.trained.keys()
        )
    )

    model = st.session_state.trained[
        selected
    ]

    prediction = st.session_state.predictions[
        selected
    ]

    # =========================
    # CONFUSION MATRIX
    # =========================

    st.subheader(
        "🔲 Confusion Matrix"
    )

    labels = np.arange(
        len(diseases)
    )

    cm = confusion_matrix(
        st.session_state.y_test,
        prediction,
        labels=labels
    )

    fig, ax = plt.subplots(
        figsize=(
            max(6, len(diseases)),
            max(5, len(diseases))
        )
    )

    ax.imshow(cm)

    ax.set_xticks(
        range(len(diseases))
    )

    ax.set_yticks(
        range(len(diseases))
    )

    ax.set_xticklabels(
        diseases,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(
        diseases
    )

    ax.set_xlabel(
        "Predicted Disease"
    )

    ax.set_ylabel(
        "Actual Disease"
    )

    ax.set_title(
        "Disease Confusion Matrix"
    )

    for i in range(len(diseases)):

        for j in range(len(diseases)):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    plt.tight_layout()

    st.pyplot(fig)

    # =========================
    # PATIENT PREDICTION
    # =========================

    st.header(
        "👤 Patient Disease Prediction"
    )

    patient = {}

    columns = st.columns(2)

    for i, column in enumerate(
        X.columns
    ):

        with columns[i % 2]:

            if column in numeric:

                values = pd.to_numeric(
                    X[column],
                    errors="coerce"
                )

                minimum = float(
                    values.min()
                )

                maximum = float(
                    values.max()
                )

                default = float(
                    values.median()
                )

                if minimum == maximum:

                    patient[column] = st.number_input(
                        column,
                        value=default
                    )

                else:

                    patient[column] = st.number_input(
                        column,
                        min_value=minimum,
                        max_value=maximum,
                        value=default
                    )

            else:

                choices = (
                    X[column]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                if len(choices) <= 30:

                    patient[column] = st.selectbox(
                        column,
                        sorted(choices)
                    )

                else:

                    patient[column] = st.text_input(
                        column
                    )

    # =========================
    # PREDICT
    # =========================

    if st.button(
        "🔮 PREDICT DISEASE",
        use_container_width=True
    ):

        patient_df = pd.DataFrame(
            [patient]
        )

        patient_df = patient_df[
            X.columns
        ]

        try:

            result = model.predict(
                patient_df
            )

            disease = diseases[
                int(result[0])
            ]

            st.success(
                f"🩺 Predicted Disease: **{disease}**"
            )

            if hasattr(
                model,
                "predict_proba"
            ):

                probability = model.predict_proba(
                    patient_df
                )[0]

                probability_df = pd.DataFrame({

                    "Disease": diseases,

                    "Probability (%)":
                        np.round(
                            probability * 100,
                            2
                        )
                }).sort_values(
                    "Probability (%)",
                    ascending=False
                )

                st.subheader(
                    "📊 Prediction Probability"
                )

                st.dataframe(
                    probability_df,
                    use_container_width=True,
                    hide_index=True
                )

        except Exception as e:

            st.error(
                f"Prediction error: {e}"
            )

# =========================
# FOOTER
# =========================

st.divider()

st.caption(
    "🩺 AI Disease Prediction System • "
    "Fast Machine Learning • Educational Use Only"
)