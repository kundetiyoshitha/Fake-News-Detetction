 FAKE NEWS DETECTION
 Hey there! Welcome to the **Fake News Detection App**, a neat tool to spot fake news headlines using machine learning. Built with **Streamlit** for a slick web interface, **scikit-learn** for a Passive-Aggressive Classifier + TF-IDF vectorizer, and **pandas** for data handling, it labels headlines as **FAKE** or **REAL**. Run it on your machine, deploy it online, or share it to analyze news with ML!

## What’s This About?
Launched on October 2, 2025, this app uses Python 3.13.7 (3.8+ works). You:
- Enter a news headline.
- Get a FAKE or REAL prediction from a pre-trained model.
- Run it locally with Streamlit or host it online for others.

The project lives in `FakeNewsDetection` with `app.py` and two model files: `tfidf_vectorizer.pkl` and `Passive-Aggressive_model.pkl`.

## What You Need
- **Python**: 3.8 or higher.
- **VS Code**: Or any code editor.
- **Git**: To clone this repo.
- **Libraries**: `streamlit`, `scikit-learn`, `pandas`.

## Setup
1. **Clone the Repo**:
   ```bash
   git clone https://github.com/your-username/FakeNewsDetection.git
   cd FakeNewsDetection
   ```

2. **Install Libraries**:
   In your terminal:
   ```bash
   pip install streamlit
   pip install scikit-learn
   pip install pandas
   ```

3. **Check Python**:
   Ensure 3.8+:
   ```bash
   python --version
   ```

4. **Folder Check**:
   Verify these files are in `FakeNewsDetection`:
   - `app.py` (main script)
   - `tfidf_vectorizer.pkl` (TF-IDF model)
   - `Passive-Aggressive_model.pkl` (classifier model)

## Running the App
### Run Locally
1. **Open the Project**:
   - Open VS Code.
   - Go `File → Open Folder` → `FakeNewsDetection` (e.g., `Desktop/FakeNewsDetection`).

2. **Start It**:
   - Open a terminal in VS Code (`Terminal → New Terminal`).
   - Navigate if needed:
     ```bash
     cd Desktop/FakeNewsDetection
     ```
   - Run:
     ```bash
     streamlit run app.py
     ```

3. **Access It**:
   - Visit `http://localhost:8501` in your browser.
   - Enter a headline, submit, and check if it’s FAKE or REAL!

4. **Stop It**:
   - Press `Ctrl + C` in the terminal.

### Deploy Online
Share it with the world via **Streamlit Cloud**, **Heroku**, or **AWS**:
- Push to GitHub.
- Add a `requirements.txt`:
  ```
  streamlit
  scikit-learn
  pandas
  ```
- Upload `app.py` and `.pkl` files to the platform.
- Follow their deployment steps for a public URL.

### Share on Local Network
Run locally but let others on your Wi-Fi try:
```bash
streamlit run app.py --server.address 0.0.0.0
```
Share your machine’s IP (e.g., `http://192.168.x.x:8501`).

## How the Model Was Trained
We used a **Passive-Aggressive Classifier** with a **TF-IDF vectorizer** on 10 fake and 10 real news headlines in Google Colab, saving `tfidf_vectorizer.pkl` and `Passive-Aggressive_model.pkl`. To retrain:

```python
!pip install scikit-learn pandas

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

# Fake and real news
fake_news = [
    "Breaking: Aliens land in New York City, President makes statement",
    "Scientists discover chocolate cures all diseases overnight",
    # ... (8 more fake headlines)
]
real_news = [
    "Stock market closes higher after positive economic data",
    "New technology breakthrough improves solar panel efficiency",
    # ... (8 more real headlines)
]

# Combine and label
texts = fake_news + real_news
labels = ['FAKE'] * len(fake_news) + ['REAL'] * len(real_news)
df = pd.DataFrame({'text': texts, 'label': labels})

# Split and vectorize
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_tfidf, y_train)

# Check accuracy
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Save models
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('Passive-Aggressive_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Download (in Colab)
from google.colab import files
files.download('tfidf_vectorizer.pkl')
files.download('Passive-Aggressive_model.pkl')
```

## Troubleshooting
- **“streamlit: command not found”**:
  - Run: `pip install streamlit`.
- **“No module named 'sklearn'”**:
  - Run: `pip install scikit-learn`.
- **“FileNotFoundError: tfidf_vectorizer.pkl”**:
  - Ensure both `.pkl` files are in `FakeNewsDetection`.
- **“Port already in use”**:
  - Try: `streamlit run app.py --server.port 8502`.
- **Pickle issues**:
  - Python version mismatch (Colab: 3.12.11, local: 3.13.7)? Save in Colab with `pickle.dump(obj, file, protocol=4)` and load locally with `pickle.load(file, encoding='latin1')`.

## Tips to Improve
- **More Data**: The model uses 20 headlines. Add more for better accuracy.
- **Model Tweaks**: Adjust `max_features` in `TfidfVectorizer` or `max_iter` in `PassiveAggressiveClassifier`.
- **Share It**: Use `ngrok` for a public URL (we chatted about this before).
- **Update Libraries**: Run `pip install --upgrade streamlit scikit-learn pandas`.

## Contributing
Got ideas? Fork the repo, tweak it, and send a pull request. Let’s make it awesome!

## License
MIT License—use it, modify it, share it, just give credit!

---

Built with Python and ML vibes. Questions? Drop them in GitHub Issues!
