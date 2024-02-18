"""
GPT-2, like other transformer models, operates on tokens rather than raw words. 
This means that a word could be split into multiple tokens, and each token will have its own embedding.
The embeddings extracted this way represent contextual word embeddings, meaning the same word can have different embeddings based on its context within different sentences.
This example uses the last hidden states as word embeddings. 
Depending on your specific use case, you might want to explore using other layers or aggregating them in some way.
"""
from transformers import GPT2Model, GPT2Tokenizer

def get_model_size(model):
    # Get the number of parameters
    num_parameters = model.num_parameters()
    # Convert to millions for readability
    size_in_millions = num_parameters / 1e6
    return size_in_millions


def get_word_embeddings(sentence):
    # Initialize the tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2')

    # Check the size of the GPT-2 model
    model_size = get_model_size(model)
    print(f"The GPT-2 model has approximately {model_size:.2f} million parameters.")

    # Encode the sentence
    inputs = tokenizer(sentence, return_tensors='pt')

    # Display the tokens
    tokens = tokenizer.tokenize(sentence)
    print("Tokens:", tokens)

    # Get model outputs
    outputs = model(**inputs)

    # Extract last hidden states
    last_hidden_states = outputs.last_hidden_state

   # The length of embeddings is the size of the last dimension of last_hidden_states
    embedding_length = last_hidden_states.shape[-1]

    # Map tokens to their embeddings
    embeddings = {}
    for idx, token in enumerate(inputs['input_ids'][0]):
        # Convert token ID back to word
        word = tokenizer.decode([token])
        # Map word to its embedding (note: embeddings are tensors)
        embeddings[word] = last_hidden_states[0, idx]

    return embeddings, embedding_length

# Example usage
sentence = "The sky is blue."
embeddings, embedding_length = get_word_embeddings(sentence)
print(f"Embedding length: {embedding_length}")
print(embeddings)

