# VerbSimplifyAI

A full-stack application that simplifies complex verbs in English sentences using word embeddings (GloVe) and a neural network model deployed via a Flask API, with a lightweight front-end for instant interactive use.

---

## 🚀 Features

- **Complex Verb Detection**  
  Uses syllable counting and a curated list of simple verbs to identify complex verbs in input sentences.

- **Embedding-Based Simplification**  
  Generates simplified verb suggestions by mapping GloVe 300d embeddings through a TensorFlow model and finding the closest simple verb.

- **Neural Model Training**  
  `train_ml.py` trains a custom neural network on pairs of complex-to-simple verb mappings.

- **Interactive Front‑End**  
  Single-page HTML/CSS/JavaScript app (`index.html`, `styles.css`, `task_embedding.js`) for real‑time sentence input and simplification.

- **RESTful API**  
  Flask-based endpoint (`/simplify_sentence`) implemented in `python_api.py` with CORS support for cross‑origin calls.

- **Production‑Ready Deployment**  
  WSGI entrypoint in `wsgi.py` for easy deployment behind any WSGI server.

---

## 🗂️ Repository Structure

```
.
├── index.html               # Front‑end HTML
├── styles.css               # Front‑end styling
├── task_embedding.js        # Front‑end logic and API calls
├── python_api.py            # Flask API server implementation
├── train_ml.py              # Script to train and save the verb simplifier model
├── wsgi.py                  # WSGI entrypoint for production
├── requirements.txt         # Python dependencies
├── verb_simplifier_model.keras  # (Generated) Trained model file
└── README.md                # This documentation
```

---

## 🛠️ Installation & Setup

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/VerbSimplifyAI.git
   cd VerbSimplifyAI
   ```

2. **Create a Python virtual environment**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Python dependencies**  
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Download spaCy model**  
   ```bash
   python -m spacy download en_core_web_sm
   ```

---

## 📈 Training the Model

> Skip this step if you already have `verb_simplifier_model.keras`.

```bash
python train_ml.py
```

This will:
- Load GloVe embeddings via `gensim`.
- Train a multi-layer neural network on predefined complex→simple verb pairs.
- Save the trained model as `verb_simplifier_model.keras`.

---

## 🚀 Running the API Server

```bash
# Development:
python wsgi.py
# Or directly:
python python_api.py
```

By default, the Flask server runs on https://localhost:5000 with SSL (`cert.pem`/`key.pem`).  
The key endpoint is:  
```
POST https://localhost:5000/simplify_sentence
Content-Type: application/json

{ "sentence": "The engineers will demonstrate the prototype." }
```

Response:
```json
{
  "simplified_sentence": "The engineers will show the prototype.",
  "changed_words": [["demonstrate","show"]]
}
```

---

## 🌐 Front-End Usage

1. Open `index.html` in your browser.
2. Enter any sentence in the input field.
3. Click **Simplify Sentence** to see the original vs. simplified output.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!  
Please fork the repository and create a pull request with descriptive details.

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
