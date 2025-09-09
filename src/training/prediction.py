from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = './bangla_model'  # or './english_model'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# 2. Prepare a new post
new_post = "ডেঙ্গু প্রতিরোধে নতুন প্রতিষেধক আসছে"

# 3. Tokenize
inputs = tokenizer(new_post, truncation=True, padding='max_length', max_length=128, return_tensors='pt')

# 4. Predict
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

# 5. Show result
if predicted_class == 1:
    print("This post contains a claim (potential fake news).")
else:
    print("This post does not contain a claim.")
