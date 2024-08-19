from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cosine
import gensim.downloader as api
import re
import spacy
from difflib import get_close_matches


app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})

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
    "plan", "arrive", "depart", "answer", "talk", "explain", "show", "watch",
    "play", "fix", "use", "borrow", "lend", "borrow", "send", "receive",
    "invite", "reply", "shout", "whisper", "yell", "ask", "answer", "teach",
    "learn", "draw", "color", "build", "destroy", "hide", "find", "play",
    "repair", "cook", "clean", "buy", "sell", "throw", "catch", "cut", "grow",
    "sleep", "wake", "dream", "eat", "drink", "run", "walk", "jump", "sit",
    "stand", "fall", "stay", "leave", "hold", "drop", "give", "take", "lose",
    "win", "speak", "listen", "drive", "stop", "start", "begin", "end", "open",
    "close", "look", "see", "hear", "touch", "taste", "smell", "feel", "run",
    "jump", "laugh", "cry", "smile", "think", "know", "believe", "hope", "plan",
    "forget", "remember", "learn", "study", "teach", "work", "play", "relax",
    "read", "write", "count", "talk", "say", "tell", "ask", "answer", "help",
    "try", "change", "keep", "lose", "find", "break", "fix", "grow", "cut",
    "build", "destroy", "fly", "drive", "walk", "run", "swim", "jump", "throw",
    "catch", "play", "rest", "travel", "cook", "eat", "drink", "sleep", "wake",
    "dream", "sit", "stand", "fall", "hold", "drop", "stop", "start", "begin",
    "end", "open", "close", "look", "watch", "see", "hear", "listen", "touch",
    "taste", "smell", "feel", "think", "know", "believe", "hope", "plan",
    "forget", "remember", "study", "learn", "teach", "work", "play", "rest", "ensure", "improve"
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
    if condition_verb.endswith('ies') and verb.endswith(('ay', 'ey', 'iy', 'oy', 'uy')):  # Handles consonant + y
        return verb[:-1] + 'ies'
    elif condition_verb.endswith('es') and verb.endswith(('sh', 'ch', 's', 'z', 'x')):  # Handles sibilant sounds
        return (verb + 'es')
    elif condition_verb.endswith('e'):
        return (verb)
    elif condition_verb.endswith('ing') and verb.endswith('ee'):
        return (verb + 'ing')
    elif condition_verb.endswith('ing') and verb.endswith('ie'):
        return (verb[:-2] + 'ying')
    elif condition_verb.endswith('ing') and verb.endswith(('w', 'x', 'y')):
        return (verb + 'ing')
    elif condition_verb.endswith('ing') and verb.endswith('ic'):
        return (verb + 'king')
    elif condition_verb.endswith('ing') and verb.endswith(('a', 'e', 'i', 'o', 'u')):
        return (verb[:-2] + 'ying')
    elif condition_verb.endswith('ing') and verb.endswith('e'):
        return (verb[:-1] + 'ing')
    elif condition_verb.endswith('ing'):
        print("here - ing condition")
        return (verb + 'ing')
    else:
        return (verb + 's')





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





@app.route('/simplify_sentence', methods=['OPTIONS','POST'])
def simplify_sentence():
    if request.method == 'OPTIONS':
        return build_cors_preflight_response()

    sentence = request.json['sentence']
    doc = nlp(sentence)
    simplified_sentence = sentence
    
    for token in doc:
        if token.pos_ == "VERB":
            lemma = token.lemma_
            embedding = get_embedding(lemma)
            print("verb found", lemma)
            if np.any(embedding) and lemma not in simple_verbs:  # Check if it's not the zero vector
                print("embedding found for ", lemma)
                predicted_embedding = predict(embedding)
                closest_verb = find_closest_word(predicted_embedding)
                print("closest verb found", closest_verb)

                # Check if the closest word is a verb
                if closest_verb:
                    conjugated_verb = conjugate_stem_verb(closest_verb, token.text)
                    print("conjugated verb:", conjugated_verb)
                    closest_word_doc = nlp(conjugated_verb)

                    if closest_word_doc[0].pos_ != "VERB":
                        print("Predicted verb -", conjugated_verb, "is not a verb")
                        # If not a verb, find the most similar verb from the simple_verbs list
                        closest_verb = get_close_matches(conjugated_verb, simple_verbs, n=1, cutoff=0.8)
                        closest_verb = closest_verb[0] if closest_verb else None

                        print("Updated closest verb", closest_verb)

                    if closest_verb:
                        simplified_sentence = simplified_sentence.replace(token.text, closest_verb)
                    elif conjugated_verb:
                        simplified_sentence = simplified_sentence.replace(token.text, conjugated_verb)

                    print("replaced verb", token.text, "with", conjugated_verb)
    return jsonify({'simplified_sentence': simplified_sentence})





def build_cors_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "https://shubham-s-bhatt.github.io")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
    return response





if __name__ == '__main__':
    app.run(debug=True, ssl_context=('cert.pem', 'key.pem'))
