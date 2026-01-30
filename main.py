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
SEARCH_QUERY = 'Iran AND (Israel OR USA OR nuclear OR conflict)'
LANGUAGE = 'en'
COUNTRY = 'US'
PERIOD = '6h'
MAX_RESULTS = 10 
JSON_FILE = 'news.json'
HISTORY_FILE = 'sent_news.txt'
MAX_WORKERS = 4

# --- NLTK SETUP ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

# --- HEADERS TO LOOK LIKE CHROME ---
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Referer': 'https://www.google.com/',
    'Upgrade-Insecure-Requests': '1',
}

# --- HELPERS ---
def get_seen_urls():
    if not os.path.exists(HISTORY_FILE): return set()
    with open(HISTORY_FILE, 'r', encoding='utf-8') as f: return set(f.read().splitlines())

def append_seen_urls(new_urls):
    with open(HISTORY_FILE, 'a', encoding='utf-8') as f:
        for url in new_urls: f.write(url + '\n')

def load_news_data():
    if not os.path.exists(JSON_FILE): return []
    try:
        with open(JSON_FILE, 'r', encoding='utf-8') as f: return json.load(f)
    except: return []

def save_news_batch(new_entries):
    if not new_entries: return
    current_data = load_news_data()
    combined_data = new_entries + current_data
    combined_data = combined_data[:40] 
    with open(JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=4)
    print(f"Saved {len(new_entries)} new articles.")

def translate_text(text):
    if not text or len(text) < 2: return ""
    try:
        # Retry logic handled internally by deep_translator usually, but we keep it simple
        return GoogleTranslator(source='auto', target='fa').translate(text)
    except:
        return text

def translate_large_text(full_text):
    """
    Splits text into chunks of ~4000 chars to respect API limits.
    """
    if not full_text or len(full_text) < 50:
        return ""
    
    # Split by paragraphs
    paragraphs = full_text.split('\n\n')
    translated_parts = []
    
    chunk = ""
    for p in paragraphs:
        if len(chunk) + len(p) < 4000:
            chunk += p + "\n\n"
        else:
            # Translate current chunk
            translated_parts.append(translate_text(chunk))
            chunk = p + "\n\n"
            time.sleep(0.5) # Be polite to API
            
    # Translate final chunk
    if chunk:
        translated_parts.append(translate_text(chunk))
            
    return '\n'.join(translated_parts)

def resolve_and_fetch(url):
    """
    1. Unmasks Google Redirects.
    2. Downloads HTML using strong headers.
    """
    try:
        # This session handles cookies and redirects like a browser
        session = requests.Session()
        session.headers.update(HEADERS)
        
        response = session.get(url, timeout=10, allow_redirects=True)
        
        if response.status_code != 200:
            return None, None
            
        return response.url, response.text
    except Exception as e:
        print(f"   > Download Error: {e}")
        return None, None

def extract_and_process(entry):
    gnews_url = entry.get('url')
    title = entry.get('title')
    publisher = entry.get('publisher', {}).get('title', 'Source')
    date = entry.get('published date')

    print(f"Processing: {title[:40]}...")

    # 1. FETCH HTML MANUALLY
    final_url, html_content = resolve_and_fetch(gnews_url)

    if not html_content:
        print("   > Skipped (Download failed)")
        return None

    # 2. PARSE WITH NEWSPAPER3K
    try:
        article = Article(final_url)
        article.set_html(html_content) # Inject the HTML we downloaded
        article.parse()
        article.nlp()
        
        summary_en = article.summary
        full_text_en = article.text
    except Exception as e:
        print(f"   > Parse Error: {e}")
        summary_en = ""
        full_text_en = ""

    # Clean data
    summary_en = summary_en.replace('\n', ' ').strip()
    
    # FALLBACK: If Full Text is empty (common on video/gallery pages), use summary
    if len(full_text_en) < 100:
        if len(summary_en) > 50:
            full_text_en = summary_en
        else:
            # If both are empty, use description from GNews
            full_text_en = entry.get('description', 'Content unavailable.')

    # 3. TRANSLATE
    # Translate Title
    title_fa = translate_text(title)
    
    # Translate Summary (Keep it short)
    if len(summary_en) > 600: summary_en = summary_en[:600] + "..."
    summary_fa = translate_text(summary_en)
    
    # Translate Full Text (Check length to avoid waiting forever)
    if len(full_text_en) > 8000: 
        full_text_en = full_text_en[:8000] + " \n[ادامه مطلب در منبع اصلی...]"
    
    full_text_fa = translate_large_text(full_text_en)

    return {
        "title_fa": title_fa,
        "summary_fa": summary_fa,
        "full_text_fa": full_text_fa,
        "title_en": title,
        "url": final_url,
        "source": publisher,
        "date": date
    }

def main():
    print("Starting Iran Radar (Fix Mode)...")
    
    google_news = GNews(language=LANGUAGE, country=COUNTRY, period=PERIOD, max_results=MAX_RESULTS)
    results = google_news.get_news(SEARCH_QUERY)
    
    seen = get_seen_urls()
    to_process = []
    
    # Filter seen URLs
    for x in results:
        # GNews URLs change slightly, so we check if we processed this Title before
        # (A slight hack, but reliable for GNews RSS)
        if x.get('url') not in seen:
            to_process.append(x)

    if not to_process:
        print("No new articles found.")
        return

    processed = []
    processed_urls = []

    # Sequential processing is safer for debugging connection issues, 
    # but we will use 2 workers to keep it reasonably fast.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(extract_and_process, entry): entry for entry in to_process}
        for future in concurrent.futures.as_completed(futures):
            try:
                res = future.result()
                if res:
                    processed.append(res)
                    processed_urls.append(res['url'])
            except Exception as e:
                print(f"Worker Error: {e}")

    save_news_batch(processed)
    
    # We save the Original GNews URL to history to avoid reprocessing
    append_seen_urls([x.get('url') for x in to_process if x.get('url')])

if __name__ == "__main__":
    main()
