from flask import Flask, render_template, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)

# Load model from current directory
tokenizer = T5Tokenizer.from_pretrained(".")
model = T5ForConditionalGeneration.from_pretrained(".")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    input_text = "summarize: " + data.get('text', '')

    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(input_ids, max_length=100, min_length=10, num_beams=4, do_sample=False, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return jsonify({'summary': summary})

if __name__ == "__main__":
    app.run(debug=True)
