# 🔍 Fake News Detection System

A web app I made to help spot fake news articles using machine learning. It's mainly for students, but honestly anyone can use it.

## What This Does

You paste in a news article or social media post, and it tells you whether it looks legit or sketchy. The model I trained got about 99.5% accuracy on the test data, which was pretty cool.

The app also points out red flags like clickbait phrases, overly emotional language, and other stuff that fake news tends to have.

## How to Use It

Just go to the app, paste your article in the text box, and hit analyze. You'll get:
- A verdict (real or fake)
- Confidence score
- Any suspicious patterns it found
- Some tips on what to do next

Pretty straightforward.

## Running It Yourself

If you want to run this locally on your computer:

**1. Clone this repo**
```bash
git clone https://github.com/yourusername/Fake-News-Detection.git
cd Fake-News-Detection
```

**2. Install the packages**
```bash
pip install -r requirements.txt
```

**3. Run it**
```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501` and you're good to go.

## What's Inside

- `app.py` - the main app code
- `fake_news_model.pkl` - the trained model (Linear SVM)
- `tfidf_vectorizer.pkl` - converts text into numbers the model understands
- `requirements.txt` - all the Python packages you need

## Tech Stack

Built with Python and Streamlit. The model is a Linear SVM trained on about 45,000 articles. I used TF-IDF for feature extraction and did some aggressive text cleaning to get better results.

The preprocessing removes stopwords, numbers, and common fake news phrases like "share this story" or "you won't believe." Turns out those patterns are super predictive.

## How I Trained It

I combined two datasets - one with fake news and one with real news. Then I:
1. Cleaned the text pretty aggressively 
2. Used TF-IDF to extract features (kept 25,000 of them)
3. Trained a Linear SVM
4. Tested with 10-fold cross validation

The aggressive cleaning helped a lot. When I just did basic preprocessing, the accuracy was lower because the model was picking up on formatting quirks instead of actual content.

## Files You Need

Make sure you have these files in your folder:
- app.py (the main code)
- fake_news_model.pkl (the AI model - might be large)
- tfidf_vectorizer.pkl (text processor)
- requirements.txt (dependencies list)

The .pkl files are the trained model components. They're a bit big (GitHub might complain), so you might need Git LFS if you're uploading them.

## Some Notes

This tool is meant to be a helper, not the final word. Always check multiple sources and use your own judgment. The model works well on the kind of articles it was trained on, but it's not magic.

Also, fake news detection is tricky because:
- Language evolves fast
- New patterns emerge constantly  
- Context matters a lot
- Sometimes real news sounds crazy (because reality is wild)

So yeah, use this as one tool among many for checking facts.

## Want to Improve It?

Feel free to fork this and make it better. Some ideas:
- Add more models and let users compare
- Include source credibility checking
- Add a Chrome extension version
- Train on more recent data
- Support
