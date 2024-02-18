from transformers import GPT2Model, GPT2Tokenizer
import torch
import pickle
import json

def get_color_word_embedding(sentence, color_word):
    #gpt-2's tokenizer adds a space before the word.
    color_word = " " + color_word
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2')

    inputs = tokenizer(sentence, return_tensors='pt')
    outputs = model(**inputs)

    last_hidden_states = outputs.last_hidden_state

    # Tokenize the sentence and the color word separately
    tokens = tokenizer.tokenize(sentence)
    color_word_tokens = tokenizer.tokenize(color_word)

    # Find the start index of the color word's tokens in the sentence tokens
    start_index = None
    for i in range(len(tokens)):
        if tokens[i:i+len(color_word_tokens)] == color_word_tokens:
            start_index = i
            break

    if start_index is None:
        raise ValueError(f"'{color_word}' not found in the sentence.")

    # Convert start index in tokens to ids index in input_ids
    ids = inputs['input_ids'][0]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    color_word_start_id = token_ids[start_index]

    start_position = (ids == color_word_start_id).nonzero(as_tuple=True)[0]

    # Assuming the first occurrence is what we're interested in
    if len(start_position) > 0:
        # Average over the sub-tokens if the color word was split into multiple tokens
        color_word_embedding = last_hidden_states[0, start_position[0]:start_position[0]+len(color_word_tokens)].mean(dim=0)
    else:
        raise ValueError(f"'{color_word}' start position not found in the input IDs.")

    return color_word_embedding


# Predefined list of color words to look for
color_words = ['blue', 'orange', 'red', 'grey', 'white', 'black', 'yellow', 'purple', 'green']

# a dictionary of colors and embeddings
color_embeddings = {}

# Generate embeddings
# space is combined with color words to form a single token
#embedding_blue = get_color_word_embedding("The sky is blue.", " blue")
#print("blue embedding:", embedding_blue)

# Loop through each color word to compute and print its embedding
for color in color_words:
    sentence = f"The sky is {color}."
    embedding = get_color_word_embedding(sentence, color)
    color_embeddings[color] = embedding
    print(f"{color} embedding:", embedding)

# Save to file
with open('color_embeddings.pkl', 'wb') as f:
    pickle.dump(color_embeddings, f)

print("Color embeddings have been saved.")

# Save the dictionary into a JSON file
# the JSON format does not natively support PyTorch tensors. 
# To save tensors in a JSON file, you need to first convert them into a format that JSON can handle, such as lists.
color_embeddings_list = {}

for color, embedding in color_embeddings.items():
    # Assuming 'embedding' is a PyTorch tensor, convert it to a list
    color_embeddings_list[color] = embedding.tolist()

with open('color_embeddings.json', 'w') as json_file:
    json.dump(color_embeddings_list, json_file, indent=4)

print("The dictionary has been saved in a human-readable JSON file.")
"""
embedding_black = get_color_word_embedding("The sky is black.", " black")
print("black embedding:", embedding_black)

# Compute the difference between the two embeddings
embedding_difference = embedding_blue - embedding_black

# Compute the absolute differences in each dimension
absolute_differences = torch.abs(embedding_difference)
# Sort the absolute differences in descending order
sorted_absolute_differences = torch.sort(absolute_differences, descending=True)[0]

print("sorted absolute differences:", sorted_absolute_differences)

# Identify dimensions with significant differences
significant_dims = torch.abs(embedding_difference).argsort(descending=True)
print("significant dims:", significant_dims)
"""
