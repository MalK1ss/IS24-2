import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils.multiclass import unique_labels

st.title("Сравнение алгоритмов машинного обучения")

uploaded_file = st.file_uploader("Загрузите CSV с данными", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Колонки в датасете:", df.columns.tolist())

    if df.isnull().values.any():
        st.warning("Обнаружены пропущенные значения! Они будут автоматически заполнены.")

    # Определение приоритетного целевого столбца
    def suggest_target_column(df):
        if "condition" in df.columns:
            return "condition"
        candidates = [(col, df[col].nunique()) for col in df.columns if 2 <= df[col].nunique() <= 10]
        if candidates:
            return sorted(candidates, key=lambda x: x[1])[0][0]
        return df.columns[0]

    suggested_target = suggest_target_column(df)

    target = st.selectbox(
        "Выберите целевой столбец (метку класса)",
        df.columns,
        index=df.columns.get_loc(suggested_target)
    )

    X = df.drop(columns=[target])
    y = df[target]

    # Обработка категориальных признаков
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    if st.button("Обучить и сравнить модели"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Импутация NaN
        imputer = SimpleImputer(strategy='mean')
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        # Масштабирование
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Список моделей
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
            "Random Forest": RandomForestClassifier(),
            "SVM": SVC()
        }

        results = []
        labels = unique_labels(y_test)

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)

            if len(labels) == 2:
                f1_0 = f1_score(y_test, y_pred, pos_label=labels[0], zero_division=0)
                f1_1 = f1_score(y_test, y_pred, pos_label=labels[1], zero_division=0)
            else:
                f1_0 = "-"
                f1_1 = "-"

            f1_avg = f1_score(y_test, y_pred, average="macro", zero_division=0)

            results.append({
                "Модель": name,
                "Accuracy": round(acc, 3),
                "F1-0": f1_0 if f1_0 == "-" else round(f1_0, 3),
                "F1-1": f1_1 if f1_1 == "-" else round(f1_1, 3),
                "Среднее F1": round(f1_avg, 3)
            })

        results_df = pd.DataFrame(results)

        st.subheader("Результаты сравнения моделей")
        st.dataframe(results_df)

        st.subheader("График Accuracy моделей")
        st.bar_chart(results_df.set_index("Модель")["Accuracy"])

else:
    st.info("Пожалуйста, загрузите CSV-файл с данными.")
