from flask import Flask, request, jsonify
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy

# Load the spacy model
nlp = spacy.load("en_core_web_sm")

# Define the chatbot's knowledge base
knowledge_base = {
    "faq": {
        "What is this website about?": "This website is about providing information on various topics.",
        "How do I contact the website owner?": "You can contact the website owner through the contact form."
    },
    "definitions": {
        "AI": "Artificial intelligence refers to the simulation of human intelligence in machines.",
        "Machine Learning": "Machine learning is a subset of AI that involves training machines to learn from data."
    },
    "suggestions": {
        "related_content": ["Article 1", "Article 2", "Article 3"]
    }
}

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.get_json()["message"]
    tokens = word_tokenize(user_input)
    tokens = [token.lower() for token in tokens if token.isalpha()]
    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token not in stop_words]

    # Check if the user is asking a question
    if tokens[0] in ["what", "how", "when", "where", "why"]:
        # Check if the question is in the knowledge base
        question = " ".join(tokens)
        if question in knowledge_base["faq"]:
            response = knowledge_base["faq"][question]
        else:
            response = "I'm not sure I understand your question."
    # Check if the user is asking for a definition
    elif tokens[0] == "define":
        term = " ".join(tokens[1:])
        if term in knowledge_base["definitions"]:
            response = knowledge_base["definitions"][term]
        else:
            response = "I'm not familiar with that term."
    # Check if the user is asking for suggestions
    elif tokens[0] == "suggest":
        response = "You might be interested in the following articles: " + ", ".join(knowledge_base["suggestions"]["related_content"])
    else:
        response = "I'm not sure I understand your request."

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)