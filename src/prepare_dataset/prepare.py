import pandas as pd


# ---------- Bangla News (BanFakeNews) ----------
def prepare_bangla_news(input_path, output_path, sample_size=500):
    df = pd.read_csv(input_path)  # e.g. ban_fake_news.csv
    df = df.rename(columns={"article": "text", "label": "label"})
    df = df.sample(n=min(sample_size, len(df)), random_state=42)
    df.to_csv(output_path, index=False)
    print(f"✅ Bangla News saved: {output_path} ({len(df)} rows)")


# ---------- English News (labeled dataset from Kaggle) ----------
def prepare_english_news(input_path, output_path, sample_size=500):
    df = pd.read_csv(input_path)
    df = df.rename(columns={"text": "text", "label": "label"})
    df = df.sample(n=min(sample_size, len(df)), random_state=42)
    df.to_csv(output_path, index=False)
    print(f"✅ English News saved: {output_path} ({len(df)} rows)")


# ---------- Bangla Facebook Posts (Fact-Checked Facebook Corpus) ----------
def prepare_bangla_fb(input_path, output_path, sample_size=500):
    df = pd.read_csv(input_path)
    df = df.rename(columns={"post_text": "text", "label": "label"})
    df = df.sample(n=min(sample_size, len(df)), random_state=42)
    df.to_csv(output_path, index=False)
    print(f"✅ Bangla FB Posts saved: {output_path} ({len(df)} rows)")


# ---------- English Facebook Posts (i will manually annotated later) ----------
def prepare_english_fb(input_path, output_path, sample_size=500):
    df = pd.read_csv(input_path)
    df = df.rename(columns={"post_text": "text", "label": "label"})
    df = df.sample(n=min(sample_size, len(df)), random_state=42)
    df.to_csv(output_path, index=False)
    print(f"English FB Posts saved: {output_path} ({len(df)} rows)")


if __name__ == "__main__":
    prepare_bangla_news("ban_fake_news.csv", "bangla_news.csv")
    prepare_english_news("english_fake_news.csv", "english_news.csv")
    prepare_bangla_fb("fact_checked_fb.csv", "bangla_fb_posts.csv")
    prepare_english_fb("english_fb_annotated.csv", "english_fb_posts.csv")
