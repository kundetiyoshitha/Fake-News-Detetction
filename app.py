import streamlit as st
import pickle

# Load the models
@st.cache_resource
def load_models():
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("Passive-Aggressive_model.pkl", "rb") as f:
        model = pickle.load(f)
    return vectorizer, model

# Page configuration
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# Title
st.title("📰 Fake News Detection App")
st.write("Enter a news article or headline to check if it's real or fake.")

# Load models
try:
    vectorizer, model = load_models()
    # Input box
    user_input = st.text_area(
        "Enter a news article or headline:",
        height=200,
        placeholder="Paste your news article here..."
    )
    # Prediction button
    if st.button("🔍 Check News", type="primary"):
        if user_input.strip() != "":
            # Transform input
            input_data = vectorizer.transform([user_input])
            # Predict
            prediction = model.predict(input_data)
            # Show raw prediction value
            st.write(f"Prediction value: {prediction[0]}")
            # Show result
            st.markdown("---")
            if prediction[0] == "FAKE":
                st.error("⚠️ This appears to be FAKE news")
            else:
                st.success("✅ This appears to be REAL news")
        else:
            st.warning("⚠️ Please enter some text to analyze")
except FileNotFoundError:
    st.error("❌ Error: Model files not found. Make sure both .pkl files are in the same folder.")
except Exception as e:
    st.error(f"❌ An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.caption("Fake News Detection using Machine Learning")
