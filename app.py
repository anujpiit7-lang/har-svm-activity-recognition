import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="HAR SVM", page_icon="🏃")
st.title("🏃 Human Activity Recognition with SVM")
st.write("Predicts activity from smartphone sensor data using a tuned RBF-kernel SVM (96% test accuracy).")

model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")
test = pd.read_csv("data/test.csv")

tab1, tab2 = st.tabs(["Random sample", "Upload CSV"])

with tab1:
    if st.button("Pick a random sample"):
        row = test.sample(1)
        X = row.drop(columns=["subject", "Activity"])
        pred = model.predict(scaler.transform(X))[0]
        true = row["Activity"].values[0]
        c1, c2 = st.columns(2)
        c1.metric("Predicted", pred)
        c2.metric("Actual", true)
        st.success("Correct!") if pred == true else st.error("Wrong prediction")

with tab2:
    file = st.file_uploader("Upload CSV with the 561 sensor features", type="csv")
    if file:
        df = pd.read_csv(file)
        X = df.drop(columns=["subject", "Activity"], errors="ignore")
        df["Predicted"] = model.predict(scaler.transform(X))
        st.dataframe(df[["Predicted"] + (["Activity"] if "Activity" in df else [])])

st.image("confusion_matrix.png", caption="Confusion matrix on test set")