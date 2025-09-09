from fake_news_detection_model import load_model, predict

# Load model + tokenizers (only once at startup)
model, bn_tok, en_tok = load_model("../../model/bangla_english_fake_news_classifier/model.safetensors")

# Test Bangla
text_bn = "ডেঙ্গু প্রতিরোধে নতুন প্রতিষেধক আসছে"
pred_bn, probs_bn = predict(text_bn, model, bn_tok, en_tok)
print("Bangla:", text_bn)
print("Prediction:", "Real" if pred_bn == 1 else "Fake", "Probs:", probs_bn)

# Test English
text_en = "Government announces free electricity for all citizens"
pred_en, probs_en = predict(text_en, model, bn_tok, en_tok)
print("English:", text_en)
print("Prediction:", "Real" if pred_en == 1 else "Fake", "Probs:", probs_en)
