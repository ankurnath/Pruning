from sentence_transformers import SentenceTransformer, util

sentences = ["I'm happy", "I am very very sad."]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


from datasets import load_dataset
imdb = load_dataset("imdb")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_function(examples):
   return tokenizer(examples["text"], truncation=True)
 
tokenized_train = imdb["train"].shuffle(seed=42).map(preprocess_function, batched=True)
tokenized_test = imdb["test"].shuffle(seed=42).map(preprocess_function, batched=True)

from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)



# #Compute embedding for both lists
# embedding_1= model.encode(sentences[0], convert_to_tensor=True)
# embedding_2 = model.encode(sentences[1], convert_to_tensor=True)

# print(util.pytorch_cos_sim(embedding_1, embedding_2).item())