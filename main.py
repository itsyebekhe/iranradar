import os
import time
import json
import nltk
import requests
import concurrent.futures
from gnews import GNews
from newspaper import Article, Config
from deep_translator import GoogleTranslator

# --- CONFIGURATION ---
SEARCH_QUERY = 'Iran AND (Israel OR USA OR conflict OR protests OR nuclear)'
LANGUAGE = 'en'
COUNTRY = 'US'
PERIOD = '6h'
MAX_RESULTS = 15
HISTORY_FILE = 'sent_news.txt'
JSON_FILE = 'news.json'
MAX_WORKERS = 5  # Number of simultaneous downloads/translations

# --- NLTK SETUP ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

# --- HELPERS ---

def get_seen_urls():
    if not os.path.exists(HISTORY_FILE):
        return set()
    with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
        # Use a set for O(1) lookups instead of list O(n)
        return set(f.read().splitlines())

def append_seen_urls(new_urls):
    with open(HISTORY_FILE, 'a', encoding='utf-8') as f:
        for url in new_urls:
            f.write(url + '\n')

def load_news_data():
    if not os.path.exists(JSON_FILE):
        return []
    try:
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []

def save_news_batch(new_entries):
    if not new_entries:
        return
    
    current_data = load_news_data()
    # Combine new data with old data
    combined_data = new_entries + current_data
    # Keep only the last 50 items
    combined_data = combined_data[:50]
    
    with open(JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=4)
    print(f"Saved {len(new_entries)} new articles to {JSON_FILE}")

def get_final_url(url):
    """
    Follows Google News redirects to get the actual publisher URL.
    This helps Newspaper3k parse correctly.
    """
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        return response.url
    except:
        return url

def extract_article_data(url):
    """
    Downloads and parses the article using a browser User-Agent.
    """
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
    config = Config()
    config.browser_user_agent = user_agent
    config.request_timeout = 10

    try:
        # Resolve redirect first
        final_url = get_final_url(url)
        
        article = Article(final_url, config=config)
        article.download()
        article.parse()
        article.nlp()
        return article.summary
    except Exception as e:
        # print(f"Failed to parse {url}: {e}") # Uncomment for debugging
        return None

def translate_text(text, retries=2):
    """
    Translates text with simple retry logic.
    """
    if not text: 
        return ""
        
    for _ in range(retries):
        try:
            return GoogleTranslator(source='auto', target='fa').translate(text)
        except Exception:
            time.sleep(1)
    return text  # Return original if translation fails

def process_single_entry(entry):
    """
    Worker function to process a single news entry.
    Returns the formatted dict or None if failed.
    """
    url = entry.get('url')
    title = entry.get('title')
    publisher = entry.get('publisher', {}).get('title', 'Source')
    published_date = entry.get('published date')

    print(f"Processing: {title[:50]}...")

    # 1. Summarize
    summary_en = extract_article_data(url)
    
    if not summary_en:
        # If parsing fails, use the description provided by GNews or a placeholder
        summary_en = entry.get('description', "Content unavailable for automated summary.")

    # Clean up whitespace
    summary_en = summary_en.replace('\n', ' ').strip()
    
    # Truncate if too long
    if len(summary_en) > 600:
        summary_en = summary_en[:600] + "..."

    # 2. Translate (Parallelized requests)
    title_fa = translate_text(title)
    summary_fa = translate_text(summary_en)

    return {
        "title_fa": title_fa,
        "summary_fa": summary_fa,
        "title_en": title,
        "summary_en": summary_en,
        "url": url,
        "source": publisher,
        "date": published_date
    }

def main():
    print("Starting Iran Radar (Optimized)...")
    
    google_news = GNews(language=LANGUAGE, country=COUNTRY, period=PERIOD, max_results=MAX_RESULTS)
    try:
        news_results = google_news.get_news(SEARCH_QUERY)
    except Exception as e:
        print(f"Error fetching news: {e}")
        return

    seen_urls = get_seen_urls()
    entries_to_process = []

    # Filter out already seen URLs before processing
    for entry in news_results:
        if entry.get('url') not in seen_urls:
            entries_to_process.append(entry)

    if not entries_to_process:
        print("No new articles found.")
        return

    print(f"Found {len(entries_to_process)} new articles. Processing...")

    new_news_items = []
    processed_urls = []

    # Use ThreadPoolExecutor to process articles in parallel (Faster)
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_entry = {executor.submit(process_single_entry, entry): entry for entry in entries_to_process}
        
        for future in concurrent.futures.as_completed(future_to_entry):
            try:
                result = future.result()
                if result:
                    new_news_items.append(result)
                    processed_urls.append(result['url'])
            except Exception as e:
                print(f"Worker exception: {e}")

    # Sort items to maintain time order (optional, as GNews usually sends newest first)
    # But since threads finish randomly, we might want to re-sort or just accept slightly mixed order.
    # Here we just save what we got.
    
    if new_news_items:
        save_news_batch(new_news_items)
        append_seen_urls(processed_urls)
    
    print("Done.")

if __name__ == "__main__":
    main()
