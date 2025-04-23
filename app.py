from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

# Model ve vektörizeri yükle
model_path = os.path.join("models", "model.pkl")
vectorizer_path = os.path.join("models", "vectorizer.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def home():
    text = ""
    prediction = ""

    if request.method == "POST":
        text = request.form.get("email-content")

        if text:
            vectorized = vectorizer.transform([text])
            pred = model.predict(vectorized)[0]
            prediction = "Spam" if pred == 1 else "Not Spam"

    return render_template("index.html", text=text, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
