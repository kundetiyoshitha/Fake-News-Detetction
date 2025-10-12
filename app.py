"""
Fake News Detector for Students
Fixed Preprocessing Version - Complete Code
"""

import streamlit as st
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from datetime import datetime

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Model Info Placeholder ---
# UPDATED to match your actual training: LinearSVC with aggressive cleaning
MODEL_ACCURACY = 0.9951
TRAIN_SIZE = 44898
AI_FEATURES = 25000  # Your actual max_features
MODEL_NAME = "Linear SVM"  # LinearSVC model
CV_MEAN = 0.9951

temp_model_info = {
    'accuracy': MODEL_ACCURACY,
    'train_size': TRAIN_SIZE,
    'features': AI_FEATURES,
    'model_name': MODEL_NAME,
    'cv_mean': CV_MEAN,
    'vocab_size': AI_FEATURES
}

# --- Download NLTK resources ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))

# --- EXACT AGGRESSIVE PREPROCESSING FROM COLAB ---
def minimal_clean_text(text):
    """Step 1: Minimal cleaning"""
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = ''.join([char for char in text if char not in string.punctuation])
    text = re.sub(r'\b\w\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@st.cache_data
def preprocess_text(text):
    """
    EXACT aggressive_clean_text function from your Colab training
    This is what achieved 99.51% accuracy
    """
    # 1. Minimal clean steps
    text = minimal_clean_text(text)
    
    # 2. Remove all stop words
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    # 3. Remove numerical cues (dates, counts, etc.)
    text = re.sub(r'\d+', '', text)
    
    # 4. Remove common fake news stylistic headers/footers
    text = re.sub(r'share this story|read more', '', text)
    
    # 5. Remove words shorter than 3 letters
    text = re.sub(r'\b\w{1,2}\b', '', text).strip()
    
    return text

# --- Helper Functions ---
def get_credibility_indicators(text):
    """Analyze text for credibility indicators"""
    indicators = {
        'Sensational': ['shocking', 'unbelievable', 'you wont believe', 'you won\'t believe', 'miracle', 'exposed'],
        'Clickbait': ['click here', 'what happened next', 'doctors hate', 'one weird trick'],
        'Urgent': ['breaking', 'urgent', 'alert', 'now', 'immediately'],
        'Emotional': ['outrageous', 'shocking', 'terrifying', 'amazing', 'incredible']
    }
    
    text_lower = text.lower()
    found = []
    
    for category, words in indicators.items():
        for word in words:
            if word in text_lower:
                found.append((category, word))
    
    return found

def generate_summary(text):
    """Generate a concise summary of the article"""
    sentences = text.split('.')[:3]
    summary = '. '.join(sentences).strip()
    if len(summary) > 200:
        summary = summary[:200] + "..."
    return summary

@st.cache_resource
def load_model():
    """Load model, vectorizer, and model info from local files"""
    try:
        with open('fake_news_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        try:
            with open('model_info.pkl', 'rb') as f:
                model_info = pickle.load(f)
        except FileNotFoundError:
            model_info = temp_model_info 
            st.warning("⚠️ 'model_info.pkl' not found. Using default stats.")

        return model, vectorizer, model_info
    except FileNotFoundError as e:
        st.error(f"❌ CRITICAL: Model files not found! {str(e)}")
        st.info("Please ensure 'fake_news_model.pkl' and 'tfidf_vectorizer.pkl' are in the same folder as app.py.")
        return None, None, None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, None

# Load resources
model, vectorizer, model_info = load_model()

if model is None:
    st.stop()

def pct(v):
    return f"{float(v):.1%}"

def fmt_int(v):
    return f"{int(v):,}"

# --- STREAMLIT UI ---
st.title("🔍 Fake News Detector for Students")
st.subheader("AI-powered tool to identify misinformation and verify news credibility")

st.divider()

with st.sidebar:
    st.header("📚 About This Tool")
    st.write("This AI-powered tool helps students:")
    st.write("✅ Detect fake news and misinformation")
    st.write("📊 Assess article credibility")
    st.write("🎯 Identify suspicious language patterns")
    st.write("📝 Get trustworthy summaries")
    
    st.divider()
    
    st.header("📊 Model Stats")
    st.metric("Accuracy", pct(model_info.get('accuracy', 0.9951)))
    st.metric("Training Articles", fmt_int(model_info.get('train_size', 44898)))
    st.metric("AI Features", fmt_int(model_info.get('features', 25000)))
    
    st.divider()
    
    st.subheader("💡 Tips for Students")
    st.write("• Always verify from multiple sources")
    st.write("• Check the author and publication")
    st.write("• Look for citations and references")
    st.write("• Be skeptical of sensational headlines")
    st.write("• Cross-reference with reputable news")

tab1, tab2, tab3 = st.tabs(["🔍 Analyze Article", "📖 Learn About Fake News", "📊 Statistics"])

with tab1:
    st.header("📰 Paste Your Article Here")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_input = st.text_area(
            "Article Text:",
            height=300,
            placeholder="Paste the news article or social media post you want to verify...",
            help="Enter at least 50 characters for accurate analysis"
        )
        
        char_count = len(user_input)
        word_count = len(user_input.split())
        st.caption(f"📝 Characters: {char_count} | Words: {word_count}")
    
    with col2:
        st.info("🎯 What to Check:")
        st.write("• Source credibility")
        st.write("• Author credentials")
        st.write("• Publication date")
        st.write("• Supporting evidence")
        st.write("• Emotional language")
    
    analyze_button = st.button("🔍 Analyze Article", type="primary", use_container_width=True)
    
    if analyze_button:
        if not user_input or len(user_input.strip()) < 50:
            st.warning("⚠️ Please enter at least 50 characters for accurate analysis!")
        else:
            with st.spinner("🔄 Analyzing article with AI..."):
                # FIXED: Using minimal preprocessing
                clean_text = preprocess_text(user_input)
                
                # Debug info (you can comment this out after testing)
                with st.expander("🔧 Debug Info (for testing)"):
                    st.write(f"Original length: {len(user_input)} chars")
                    st.write(f"Cleaned length: {len(clean_text)} chars")
                    st.write(f"Cleaned preview: {clean_text[:200]}...")
                
                features = vectorizer.transform([clean_text])
                prediction = model.predict(features)[0]
                
                # Confidence calculation
                if hasattr(model, 'decision_function'):
                    confidence_raw = model.decision_function(features)[0]
                    confidence_score = min(abs(confidence_raw) / 10, 0.99)
                else:
                    confidence_score = 0.85
                
                indicators = get_credibility_indicators(user_input)
                summary = generate_summary(user_input)
            
            st.divider()
            st.subheader("🎯 Analysis Results")
            
            col_res1, col_res2, col_res3 = st.columns([2, 1, 1])
            
            with col_res1:
                if prediction == 0:
                    st.error("🚨 LIKELY FAKE NEWS")
                    st.warning("⚠️ Warning: This article shows strong characteristics of misinformation or fake news.")
                    st.write("")
                    st.write("**Recommendation:**")
                    st.write("• Do NOT share this article")
                    st.write("• Verify from trusted news sources")
                    st.write("• Check fact-checking websites")
                else:
                    st.success("✅ LIKELY RELIABLE NEWS")
                    st.info("✓ Analysis: This article appears to be legitimate news content.")
                    st.write("")
                    st.write("**Still Remember:**")
                    st.write("• Cross-check with other sources")
                    st.write("• Verify key facts independently")
                    st.write("• Check author credentials")
            
            with col_res2:
                verdict_label = "FAKE" if prediction == 0 else "REAL"
                risk_label = "High Risk" if prediction == 0 else "Low Risk"
                st.metric("Verdict", verdict_label)
                if prediction == 0:
                    st.error(risk_label)
                else:
                    st.success(risk_label)
            
            with col_res3:
                st.metric("AI Confidence", f"{confidence_score:.0%}")
            
            if indicators:
                st.divider()
                st.subheader("🔍 Suspicious Language Detected")
                st.warning("The following potentially problematic patterns were found:")
                
                indicator_cols = st.columns(2)
                for idx, (category, word) in enumerate(indicators[:6]):
                    with indicator_cols[idx % 2]:
                        st.write(f"**{category}:** '{word}'")
            
            st.divider()
            st.subheader("📝 Article Summary")
            st.info(summary)
            
            with st.expander("📊 View Detailed Analysis"):
                st.write(f"**Original Text Length:** {len(user_input)} characters")
                st.write(f"**Processed Text Length:** {len(clean_text)} characters")
                st.write(f"**Word Count (Original):** {word_count} words")
                st.write(f"**Feature Vector Size:** {features.shape[1]}")
                st.write(f"**Model Used:** {model_info.get('model_name', 'Linear SVM')}")
                st.write(f"**Analysis Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            st.divider()
            st.subheader("💡 What Should You Do?")
            
            if prediction == 1:
                st.warning("⚠️ Recommended Actions:")
                st.write("1. **Stop and Verify:** Don't share this article yet")
                st.write("2. **Check Sources:** Visit fact-checking websites like Snopes, FactCheck.org")
                st.write("3. **Research:** Look for the same story on reputable news sites")
                st.write("4. **Question:** Who wrote this? What's their agenda?")
                st.write("5. **Educate:** Warn others about this potential misinformation")
            else:
                st.info("✅ Best Practices:")
                st.write("1. **Still Verify:** Cross-check key facts with other sources")
                st.write("2. **Read Fully:** Don't just rely on headlines")
                st.write("3. **Check Date:** Ensure the information is current")
                st.write("4. **Look for Bias:** Consider multiple perspectives")
                st.write("5. **Share Wisely:** Include context when sharing")

with tab2:
    st.header("📖 Understanding Fake News")
    
    col_edu1, col_edu2 = st.columns(2)
    
    with col_edu1:
        st.subheader("🎯 What is Fake News?")
        st.write("Fake news is false or misleading information presented as news. It includes:")
        st.write("")
        st.write("**Fabricated Stories:** Completely made-up content")
        st.write("**Manipulated Images:** Doctored photos or videos")
        st.write("**Misleading Headlines:** Clickbait that misrepresents content")
        st.write("**Satire Misunderstood:** Jokes taken as real news")
        st.write("**Biased Reporting:** One-sided or distorted facts")
        
        st.divider()
        
        st.subheader("🚩 Red Flags to Watch For")
        st.write("• Sensational or emotional headlines")
        st.write("• No author or source information")
        st.write("• Poor grammar and spelling")
        st.write("• Lack of dates or verifiable facts")
        st.write("• Requests to share immediately")
        st.write("• URLs that mimic real news sites")
        st.write("• No other sources reporting the story")
    
    with col_edu2:
        st.subheader("✅ How to Verify News")
        
        st.write("**1. Check the Source**")
        st.write("• Is it a known, reputable outlet?")
        st.write("• Does the site have an 'About' page?")
        st.write("• Can you find contact information?")
        st.write("")
        
        st.write("**2. Verify the Author**")
        st.write("• Is the author a real person?")
        st.write("• What are their credentials?")
        st.write("• Have they written other articles?")
        st.write("")
        
        st.write("**3. Check the Date**")
        st.write("• When was it published?")
        st.write("• Is the information still relevant?")
        st.write("• Are images/videos recent?")
        st.write("")
        
        st.write("**4. Read Beyond Headlines**")
        st.write("• Does the content match the headline?")
        st.write("• Are there credible sources cited?")
        st.write("• Is there supporting evidence?")
        st.write("")
        
        st.write("**5. Check Other Sources**")
        st.write("• Do reputable outlets report this?")
        st.write("• What do fact-checkers say?")
        st.write("• Are multiple perspectives shown?")
    
    st.divider()
    st.subheader("🔗 Trusted Fact-Checking Resources")
    
    col_res1, col_res2, col_res3 = st.columns(3)
    
    with col_res1:
        st.write("**Fact-Checking Sites:**")
        st.write("• Snopes.com")
        st.write("• FactCheck.org")
        st.write("• PolitiFact.com")
        st.write("• International Fact-Checking Network")
    
    with col_res2:
        st.write("**Media Literacy:**")
        st.write("• News Literacy Project")
        st.write("• Common Sense Media")
        st.write("• MediaWise for Teens")
        st.write("• First Draft News")
    
    with col_res3:
        st.write("**Trusted News Sources:**")
        st.write("• Associated Press (AP)")
        st.write("• Reuters")
        st.write("• BBC News")
        st.write("• NPR")

with tab3:
    st.header("📊 Model Performance & Statistics")
    
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    
    with col_stat1:
        st.metric("Model Accuracy", pct(model_info.get('accuracy', 0)))
    
    with col_stat2:
        st.metric("Training Articles", fmt_int(model_info.get('train_size', 0)))
    
    with col_stat3:
        st.metric("Test Articles", fmt_int(model_info.get('test_size', 0)))
    
    with col_stat4:
        st.metric("AI Features", fmt_int(model_info.get('features', 0)))
    
    st.divider()
    
    col_img1, col_img2 = st.columns([3, 2])
    
    with col_img1:
        st.subheader("📈 Confusion Matrix")
        try:
            st.image('confusion_matrix.png', use_container_width=True)
            st.caption("Visual representation of model's prediction accuracy")
        except:
            st.info("Confusion matrix image will appear here when 'confusion_matrix.png' is in the same directory.")
    
    with col_img2:
        st.subheader("📊 Model Details")
        st.write(f"**Model Type:** {model_info.get('model_name', 'N/A')}")
        st.write("")
        st.write("**Training Info:**")
        st.write(f"• Cross-Validation: {pct(model_info.get('cv_mean', 0))}")
        st.write(f"• Vocabulary Size: {fmt_int(model_info.get('vocab_size', 0))}")
        st.write("• Feature Extraction: TF-IDF")
        st.write("• N-gram Range: (1, 3)")
        st.write("")
        st.write("**Performance:**")
        st.write("• Optimized for student use")
        st.write("• Fast prediction time")
        st.write("• High accuracy on test set")
        st.write("• Handles various text lengths")

st.divider()
st.subheader("⚡ Powered by AI & Machine Learning")
st.caption("🎓 For Students: This tool helps you develop critical thinking skills")
st.caption("⚠️ Always verify information from multiple trusted sources before accepting it as fact")
st.caption("📚 Use this tool as a learning aid, not a replacement for critical analysis")
st.caption("© 2025 Fake News Detector | Educational Purpose Only")