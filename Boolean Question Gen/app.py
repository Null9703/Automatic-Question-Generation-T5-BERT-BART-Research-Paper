from flask import Flask, request, render_template
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Initialize Flask app
app = Flask(__name__)

# Load the model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('boolq_model')
tokenizer = T5Tokenizer.from_pretrained('boolq_tokenizer')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def beam_search_decoding(inp_ids, attn_mask):
    beam_output = model.generate(
        input_ids=inp_ids,
        attention_mask=attn_mask,
        max_length=256,
        num_beams=10,
        num_return_sequences=3,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    questions = [
        tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for out in beam_output
    ]
    return [question.strip().capitalize() for question in questions]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    passage = request.form['passage']
    truefalse = "yes"  # This can be dynamic if needed
    text = f"truefalse: {truefalse} passage: {passage} </s>"
    
    encoding = tokenizer.encode_plus(text, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
    
    questions = beam_search_decoding(input_ids, attention_masks)
    return render_template('index.html', passage=passage, questions=questions)

if __name__ == '__main__':
    app.run(debug=True)
