from flask import Flask, request, render_template
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoModelForSeq2SeqLM, BartTokenizer, BartForConditionalGeneration

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained models and tokenizers for T5, BERT, and BART-based question generation models
# T5 model and tokenizer
t5_model = T5ForConditionalGeneration.from_pretrained('t5_qgen_model')
t5_tokenizer = T5Tokenizer.from_pretrained('t5_qgen_tokenizer')

# BERT model and tokenizer (using AutoModelForSeq2SeqLM for BERT)
bert_model = AutoModelForSeq2SeqLM.from_pretrained('bert_qgen_model')
bert_tokenizer = BartTokenizer.from_pretrained('bert_qgen_tokenizer')

# BART model and tokenizer
bart_model = BartForConditionalGeneration.from_pretrained('bart_qgen_model')
bart_tokenizer = BartTokenizer.from_pretrained('bart_qgen_tokenizer')

# Function to generate questions
def get_questions(context, model_type="bart", max_length=64, temperature=0.6, top_k=30, top_p=0.85, num_beams=5):
    qns = []
    
    # Split the context into sentences
    sentences = context.split('.')
    
    # Select the appropriate model and tokenizer based on user choice
    if model_type == "t5":
        model_q = t5_model
        tokenizer_q = t5_tokenizer
    elif model_type == "bert":
        model_q = bert_model
        tokenizer_q = bert_tokenizer
    else:  # default to bart
        model_q = bart_model
        tokenizer_q = bart_tokenizer

    # Iterate over the sentences in the context
    for sentence in sentences[:-1]:
        input_text = f"answer: {''}  context: {sentence} </s>"
        features = tokenizer_q([input_text], return_tensors='pt', padding=True, truncation=True)

        # Generate questions with sampling-based decoding
        output = model_q.generate(
            input_ids=features['input_ids'],
            attention_mask=features['attention_mask'],
            max_length=max_length,
            temperature=temperature,  # Controls randomness
            top_k=top_k,              # Top-k sampling
            top_p=top_p,              # Nucleus sampling
            num_beams=num_beams,      # Beam search (adjust if you want to use it)
            do_sample=True,           # Enable sampling-based generation
            early_stopping=False      # Disable early stopping for num_beams=1
        )

        # Decode the output and ensure it's a valid question
        decoded_output = tokenizer_q.decode(output[0], skip_special_tokens=True)
        
        # For T5: Remove "question: " prefix if it exists
        if model_type == "t5" and decoded_output.startswith("question: "):
            decoded_output = decoded_output[len("question: "):]

        # For BERT: Add a "?" to the end if it doesn't already end with one
        if model_type == "bert" and not decoded_output.endswith('?'):
            decoded_output += '?'

        # Append the valid question to the list
        if decoded_output.strip():  # Ensure non-empty and relevant output
            if decoded_output.endswith('?'):
                qns.append(decoded_output)

    return qns

# Route to render the HTML form
@app.route('/')
def home():
    return render_template('index.html')

# API Route to generate questions (used when the form is submitted)
@app.route('/generate_questions', methods=['POST'])
def generate_questions():
    # Get the input context from the form
    context = request.form.get('context', '')
    model_type = request.form.get('model_type', 'bart')  # Default to 'bart'

    # Generate questions from the context
    if context:
        questions = get_questions(context, model_type=model_type)
        if questions:
            return render_template('index.html', questions=questions, context=context, model_type=model_type)
        else:
            error = "No questions could be generated from the provided context."
            return render_template('index.html', error=error, context=context, model_type=model_type)
    else:
        error = "Context is required to generate questions."
        return render_template('index.html', error=error, model_type=model_type)

if __name__ == '__main__':
    app.run(debug=True)