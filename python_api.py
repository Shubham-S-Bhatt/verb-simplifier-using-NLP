from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cosine
import gensim.downloader as api
import re
import spacy
from difflib import get_close_matches
import ssl
from waitress import serve  # Import serve from waitress

app = Flask(__name__)
CORS(app)

# Load spaCy model
nlp = spacy.load('en_core_web_sm')


# Load the trained model
model = tf.keras.models.load_model("verb_simplifier_model.keras")
print("Model loaded from verb_simplifier_model.keras")

# Load the GloVe 300-dimensional model using gensim
print("Loading GloVe 300d model...")
glove_model = api.load("glove-wiki-gigaword-300")
embedding_size = 300
print("GloVe 300d model loaded.")

# Predefined list of simple verbs
simple_verbs = [
    "do", "make", "get", "take", "use", "find", "go", "see", "run", "eat", "sleep",
    "come", "give", "put", "keep", "let", "say", "tell", "ask", "work", "play",
    "talk", "walk", "read", "write", "learn", "move", "sit", "stand", "talk",
    "look", "watch", "hear", "listen", "stop", "start", "begin", "end", "open",
    "close", "try", "help", "call", "leave", "stay", "live", "die", "love",
    "hate", "like", "dislike", "need", "want", "meet", "pay", "buy", "sell",
    "win", "lose", "grow", "cut", "break", "fix", "teach", "study", "remember",
    "forget", "follow", "lead", "push", "pull", "hold", "bring", "carry",
    "throw", "catch", "build", "destroy", "drive", "fly", "swim", "draw",
    "paint", "count", "measure", "mix", "cook", "clean", "wash", "repair",
    "jump", "climb", "sing", "dance", "laugh", "cry", "smile", "frown",
    "shout", "whisper", "play", "rest", "travel", "capture", "ensure"
]

# Utility function to count syllables
def count_syllables(word):
    word = word.lower()
    syllable_count = len(re.findall(r'[aeiouy]+', word))
    return syllable_count

# Utility function to check if a verb is complex
def is_complex_verb(verb):
    syllable_count = count_syllables(verb)
    return syllable_count > 2 and verb not in simple_verbs

# Utility function to get embeddings
def get_embedding(word):
    if is_complex_verb(word) and word in glove_model:
        return glove_model[word]
    else:
        return np.zeros(embedding_size)  # Return a zero vector if word is not complex or not found

# Prediction function
def predict(input_embedding):
    input_embedding = np.array(input_embedding).reshape(1, -1)  # Reshape for a single sample
    predicted_embedding = model.predict(input_embedding)
    return predicted_embedding.flatten().tolist()



def conjugate_stem_verb(verb, condition_verb):
    if condition_verb.endswith('ies') and verb.endswith(('ay', 'ey', 'iy', 'oy', 'uy') ):  # Handles consonant + y
        return verb[:-1] + 'ies'
    elif condition_verb.endswith('es') and verb.endswith(('sh', 'ch', 's', 'z', 'x')):  # Handles sibilant sounds
        return verb + 'es'
    else:
        return verb + 's'  # Default rule for third-person singular




# Function to find the closest word based on cosine similarity
def find_closest_word(predicted_embedding):
    closest_word = None
    smallest_distance = float('inf')
    
    for word in glove_model.index_to_key:
        embedding = glove_model[word]
        distance = cosine(predicted_embedding, embedding)
        if distance < smallest_distance:
            smallest_distance = distance
            closest_word = word

    return closest_word

@app.route('/simplify_sentence', methods=['POST'])
def simplify_sentence():
    sentence = request.json['sentence']
    doc = nlp(sentence)
    simplified_sentence = sentence
    
    for token in doc:
        if token.pos_ == "VERB":
            lemma = token.lemma_
            embedding = get_embedding(lemma)
            
            if np.any(embedding):  # Check if it's not the zero vector
                predicted_embedding = predict(embedding)
                closest_verb = find_closest_word(predicted_embedding)
                
                # Check if the closest word is a verb
                if closest_verb:
                    closest_word_doc = nlp(closest_verb)
                    if closest_word_doc[0].pos_ != "VERB":
                        # If not a verb, find the most similar verb from the simple_verbs list
                        closest_verb = get_close_matches(closest_verb, simple_verbs, n=1, cutoff=0.6)
                        closest_verb = closest_verb[0] if closest_verb else None
                    
                    if closest_verb:
                        print("original word:", token.text)
                        conjugated_verb = conjugate_stem_verb(closest_verb, token.text)
                        print("conjugated verb:", conjugated_verb)
                        simplified_sentence = simplified_sentence.replace(token.text, conjugated_verb)
    
    return jsonify({'simplified_sentence': simplified_sentence})




if __name__ == '__main__':
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile='cert.pem', keyfile='key.pem')
    serve(app, host='0.0.0.0', port=8000, ssl_context=context)
