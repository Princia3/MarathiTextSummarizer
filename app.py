from flask import Flask, render_template, request
from src_extractive import summarize_extractive
from src_preprocess import normalize_text, split_sentences

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    try:
        text = request.form["text"].strip()
        num_sentences = int(request.form.get("num_sentences", 2))

        if not text:
            return render_template("index.html", error="कृपया मजकूर प्रविष्ट करा.")
        total_sentences = len(split_sentences(normalize_text(text)))

        if num_sentences >= total_sentences:
            return render_template(
                "index.html",
                text=text,
                error=f"सारांशातील वाक्यांची संख्या ({num_sentences}) मूळ वाक्यांपेक्षा जास्त आहे. कृपया {total_sentences} पेक्षा कमी संख्या निवडा."
            )

        summary, explanation = summarize_extractive(text, num_sentences=num_sentences)

        return render_template(
            "index.html",
            text=text,
            summary=summary,
            explanation=explanation
        )
    except Exception as e:
        return render_template("index.html", error=f"त्रुटी आली: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
