from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load and preprocess the CSV data
df = pd.read_csv('chatbotdatset.csv')
df['Question'] = df['Question'].str.lower()

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(df['Question'])

def get_answer(user_question):
    user_vector = vectorizer.transform([user_question.lower()])
    similarities = cosine_similarity(user_vector, question_vectors)
    most_similar_idx = similarities.argmax()
    
    if similarities[0][most_similar_idx] > 0.5:  # Threshold for similarity
        return {
            'section': df.iloc[most_similar_idx]['Section'],
            'question': df.iloc[most_similar_idx]['Question'],
            'answer': df.iloc[most_similar_idx]['Answer'],
            'link': 'https://example.com/more-info'  # Add a link for further information
        }
    else:
        return {
            'section': 'Unknown',
            'question': user_question,
            'answer': "I'm sorry, I don't have an answer for that question.",
            'link': 'https://example.com/help'  # Add a link for help
        }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_question = request.json['question']
    response = get_answer(user_question)
    return jsonify(response)

@app.route('/get_suggestions')
def get_suggestions():
    suggestions = df['Question'].tolist()
    return jsonify({'suggestions': suggestions})

if __name__ == '__main__':
    app.run(debug=True)