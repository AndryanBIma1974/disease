import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression
)

from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor
)

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    VotingClassifier,
    VotingRegressor
)

from sklearn.neural_network import (
    MLPClassifier,
    MLPRegressor
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="AI AutoML Prediction Studio",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI AutoML Prediction Studio")
st.write(
    "Upload any suitable dataset, select the target column, "
    "and automatically train multiple machine-learning models."
)

# =========================================================
# UPLOAD DATASET
# =========================================================

st.header("📂 Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel dataset",
    type=["csv", "xlsx"]
)

if uploaded_file is None:

    st.info("👆 Please upload a CSV or Excel dataset.")

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

    st.error(f"❌ Could not load dataset: {e}")

    st.stop()

st.success(
    f"✅ Dataset uploaded successfully: {uploaded_file.name}"
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
# PREVIEW
# =========================================================

with st.expander("👀 View Dataset"):

    st.dataframe(
        data,
        use_container_width=True
    )

# =========================================================
# COLUMN INFORMATION
# =========================================================

with st.expander("🔎 Column Information"):

    info = pd.DataFrame({

        "Column": data.columns,

        "Data Type": [
            str(data[col].dtype)
            for col in data.columns
        ],

        "Missing": [
            int(data[col].isnull().sum())
            for col in data.columns
        ],

        "Unique Values": [
            int(data[col].nunique())
            for col in data.columns
        ]
    })

    st.dataframe(
        info,
        use_container_width=True
    )

# =========================================================
# TARGET SELECTION
# =========================================================

st.header("🎯 Prediction Target")

target_column = st.selectbox(
    "Select the column you want the AI to predict",
    data.columns
)

st.info(
    f"🎯 Target selected: **{target_column}**"
)

# =========================================================
# PREPARE DATA
# =========================================================

X = data.drop(
    columns=[target_column]
).copy()

y = data[target_column].copy()

# Remove missing target rows

valid = y.notna()

X = X.loc[valid].copy()
y = y.loc[valid].copy()

# =========================================================
# AUTOMATIC PROBLEM DETECTION
# =========================================================

numeric_target = pd.to_numeric(
    y.astype(str).str.replace(",", ""),
    errors="coerce"
)

numeric_ratio = numeric_target.notna().mean()

if numeric_ratio >= 0.95:

    y = numeric_target

    if y.nunique() > 10:

        problem_type = "Regression"

    else:

        problem_type = "Classification"

else:

    problem_type = "Classification"


if problem_type == "Regression":

    st.success(
        "📈 Regression detected automatically."
    )

else:

    st.success(
        "🏷️ Classification detected automatically."
    )

# =========================================================
# HANDLE FEATURES
# =========================================================

categorical_columns = X.select_dtypes(
    include=["object", "category", "bool"]
).columns

if len(categorical_columns) > 0:

    X = pd.get_dummies(
        X,
        columns=categorical_columns,
        drop_first=True
    )

# =========================================================
# HANDLE MISSING VALUES
# =========================================================

for col in X.columns:

    if X[col].isnull().any():

        if pd.api.types.is_numeric_dtype(X[col]):

            X[col] = X[col].fillna(
                X[col].median()
            )

        else:

            mode = X[col].mode()

            if len(mode) > 0:

                X[col] = X[col].fillna(
                    mode.iloc[0]
                )

            else:

                X[col] = X[col].fillna(0)

# =========================================================
# ENCODE TARGET
# =========================================================

label_encoder = None

if problem_type == "Classification":

    if not pd.api.types.is_numeric_dtype(y):

        label_encoder = LabelEncoder()

        y = label_encoder.fit_transform(
            y.astype(str)
        )

# =========================================================
# CHECK TARGET
# =========================================================

if y.nunique() < 2:

    st.error(
        "❌ Target must contain at least two different values."
    )

    st.stop()

# =========================================================
# SETTINGS
# =========================================================

st.header("⚙️ Training Settings")

col1, col2 = st.columns(2)

with col1:

    test_size = st.slider(
        "Test Data (%)",
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
# TRAIN TEST SPLIT
# =========================================================

if problem_type == "Classification":

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

else:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size / 100,
        random_state=random_state
    )

# =========================================================
# SCALING
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

if problem_type == "Classification":

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

        "MLP Neural Network":
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=1500,
                random_state=random_state
            )
    }

else:

    models = {

        "Linear Regression":
            LinearRegression(),

        "KNN":
            KNeighborsRegressor(
                n_neighbors=5
            ),

        "Random Forest":
            RandomForestRegressor(
                n_estimators=200,
                random_state=random_state
            ),

        "Extra Trees":
            ExtraTreesRegressor(
                n_estimators=200,
                random_state=random_state
            ),

        "Gradient Boosting":
            GradientBoostingRegressor(
                random_state=random_state
            ),

        "MLP Neural Network":
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                max_iter=1500,
                random_state=random_state
            )
    }

# =========================================================
# TRAIN BUTTON
# =========================================================

st.header("🧠 AI Model Training")

if st.button(
    "🚀 Train All Models",
    use_container_width=True
):

    results = []

    trained_models = {}

    predictions = {}

    progress = st.progress(0)

    total = len(models)

    for i, (name, model) in enumerate(
        models.items()
    ):

        try:

            # Tree models do not require scaling

            tree_models = [
                "Random Forest",
                "Extra Trees",
                "Gradient Boosting"
            ]

            if name in tree_models:

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

            # =============================================
            # CLASSIFICATION
            # =============================================

            if problem_type == "Classification":

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

                    "Accuracy (%)":
                        round(accuracy * 100, 2),

                    "Precision (%)":
                        round(precision * 100, 2),

                    "Recall (%)":
                        round(recall * 100, 2),

                    "F1 (%)":
                        round(f1 * 100, 2)
                })

            # =============================================
            # REGRESSION
            # =============================================

            else:

                mae = mean_absolute_error(
                    y_test,
                    prediction
                )

                mse = mean_squared_error(
                    y_test,
                    prediction
                )

                rmse = np.sqrt(mse)

                r2 = r2_score(
                    y_test,
                    prediction
                )

                results.append({

                    "Model": name,

                    "MAE":
                        round(mae, 2),

                    "MSE":
                        round(mse, 2),

                    "RMSE":
                        round(rmse, 2),

                    "R²":
                        round(r2, 4)
                })

            trained_models[name] = model

            predictions[name] = prediction

        except Exception as e:

            st.warning(
                f"⚠️ {name} failed: {e}"
            )

        progress.progress(
            (i + 1) / total
        )

    # =====================================================
    # SAVE
    # =====================================================

    results_df = pd.DataFrame(results)

    st.session_state.results = results_df
    st.session_state.models = trained_models
    st.session_state.predictions = predictions
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test
    st.session_state.problem_type = problem_type

    st.success(
        "🎉 Model training completed!"
    )

# =========================================================
# RESULTS
# =========================================================

if "results" in st.session_state:

    results_df = st.session_state.results

    st.header("🏆 Model Comparison")

    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True
    )

    # =====================================================
    # BEST MODEL
    # =====================================================

    if problem_type == "Classification":

        best_index = results_df[
            "Accuracy (%)"
        ].idxmax()

        best_score = results_df.loc[
            best_index,
            "Accuracy (%)"
        ]

        best_model = results_df.loc[
            best_index,
            "Model"
        ]

        st.success(
            f"🏆 Best Model: **{best_model}** "
            f"with **{best_score}% accuracy**"
        )

    else:

        best_index = results_df[
            "R²"
        ].idxmax()

        best_score = results_df.loc[
            best_index,
            "R²"
        ]

        best_model = results_df.loc[
            best_index,
            "Model"
        ]

        st.success(
            f"🏆 Best Model: **{best_model}** "
            f"with **R² = {best_score}**"
        )

    # =====================================================
    # PERFORMANCE GRAPH
    # =====================================================

    st.subheader("📈 Model Performance")

    fig, ax = plt.subplots()

    if problem_type == "Classification":

        ax.bar(
            results_df["Model"],
            results_df["Accuracy (%)"]
        )

        ax.set_ylabel(
            "Accuracy (%)"
        )

    else:

        ax.bar(
            results_df["Model"],
            results_df["R²"]
        )

        ax.set_ylabel(
            "R² Score"
        )

    ax.set_title(
        "AI Model Comparison"
    )

    plt.xticks(
        rotation=30
    )

    st.pyplot(fig)

    # =====================================================
    # DETAILED MODEL
    # =====================================================

    st.header("🔍 Detailed Prediction Analysis")

    selected_model = st.selectbox(
        "Choose a trained model",
        list(
            st.session_state.models.keys()
        )
    )

    model = st.session_state.models[
        selected_model
    ]

    prediction = st.session_state.predictions[
        selected_model
    ]

    y_test = st.session_state.y_test

    # =====================================================
    # CLASSIFICATION
    # =====================================================

    if problem_type == "Classification":

        st.subheader(
            "🔲 Confusion Matrix"
        )

        cm = confusion_matrix(
            y_test,
            prediction
        )

        fig, ax = plt.subplots()

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm
        )

        disp.plot(
            ax=ax
        )

        ax.set_title(
            selected_model
        )

        st.pyplot(fig)

    # =====================================================
    # REGRESSION
    # =====================================================

    else:

        st.subheader(
            "📈 Actual vs Predicted"
        )

        fig, ax = plt.subplots()

        ax.scatter(
            y_test,
            prediction
        )

        ax.set_xlabel(
            "Actual"
        )

        ax.set_ylabel(
            "Predicted"
        )

        ax.set_title(
            selected_model
        )

        st.pyplot(fig)

    # =====================================================
    # PREDICTION TABLE
    # =====================================================

    st.subheader(
        "📋 Prediction Results"
    )

    comparison = pd.DataFrame({

        "Actual":
            np.array(y_test),

        "Predicted":
            prediction
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
        "📥 Download Predictions",
        csv,
        "AI_predictions.csv",
        "text/csv"
    )