# MBTI-Personality-Prediction-from-Social-Media-posts
This project was based on predicting the MBTI personalities from a user's last 50 social media posts.
Each MBTI personality code is a combination of 4 binary personality traits i.e Introversion(I)/Extroversion(E) + Sensing(S)/Intuition(N) + Thinking(T)/Feeling(F) + Judging(J)/Perceiving(P)

Dataset Link: https://drive.google.com/file/d/1-7RuEPflyXAz5cz11oUbjPjM1rR6-u68/view?usp=sharing

Two approaches to the problem tried out:-
  1. Making a single 16 class(16 MBTI personality codes) classification model.
  2. Making an ENSEMBLE of 4 models where each model is associated with predicting the indivisual personality trait.

Features Extracted using:-
  1. CountVectorizer
  2. TF-IDF Vectorizer
  3. Word2Vec Vectorizer

Different Models Tried:-
  1. Random Forest
  2. XGBoost Classifier
  3. Multilayer Perceptron
  4. Logistic Regression
  5. Support Vector Machine Classifier
  6. ExtraTrees Classifier
  7. Decision Trees Classifier
  8. K-Nearest Neighbor Classifier

Please go through the results slides for complete analysis results and explanations.


-----------------------------------------------------------test3-----------------------------------------------------------------------------------------------
