import os
import json
import time
import logging
import requests
import urllib.parse
import concurrent.futures
from datetime import datetime
from dateutil import parser
from bs4 import BeautifulSoup
from gnews import GNews
from fake_useragent import UserAgent

# --- CONFIGURATION ---
CONFIG = {
    # Search for high-impact keywords related to Iran
    'SEARCH_QUERY': 'Iran AND (Israel OR USA OR nuclear OR conflict OR sanctions OR currency OR IRGC)',
    'LANGUAGE': 'en',
    'COUNTRY': 'US',
    'PERIOD': '4h',  # Short period to ensure we only process fresh news
    'MAX_RESULTS': 15, # Limit results to save processing time
    'FILES': {
        'NEWS': 'news.json',
        'MARKET': 'market.json',
        'HISTORY': 'seen_news.txt'
    },
    'TIMEOUT': 20,
    'MAX_WORKERS': 4, # Parallel threads
    'POLLINATIONS_KEY': os.environ.get('POLLINATIONS_API_KEY') # Secret Key
}

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

class IranNewsRadar:
    def __init__(self):
        self.ua = UserAgent()
        self.seen_urls = self._load_seen()
        self.api_key = CONFIG['POLLINATIONS_KEY']
        
        if not self.api_key:
            logger.warning("⚠️ No API Key found! AI analysis will be limited.")

    def _get_headers(self):
        """Random headers to mimic a real browser."""
        return {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Referer': 'https://www.google.com/',
            'Upgrade-Insecure-Requests': '1'
        }

    def _load_seen(self):
        if not os.path.exists(CONFIG['FILES']['HISTORY']): return set()
        with open(CONFIG['FILES']['HISTORY'], 'r', encoding='utf-8') as f:
            return set(f.read().splitlines())

    def _save_seen(self, new_urls):
        with open(CONFIG['FILES']['HISTORY'], 'a', encoding='utf-8') as f:
            for url in new_urls: f.write(url + '\n')

    # --- 1. MARKET DATA FETCH ---
    def fetch_market_rates(self):
        """Fetches USD price from AlanChand or fallback."""
        url = "https://alanchand.com/en/currencies-price/usd"
        try:
            response = requests.get(url, headers=self._get_headers(), timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                price = 0
                input_tag = soup.find('input', attrs={'data-curr': 'tmn'})
                if input_tag:
                    val = input_tag.get('data-price') or input_tag.get('value')
                    if val: price = int(int(val.replace(',', '')) / 10)
                
                if price > 0: 
                    return {"usd": f"{price:,}", "updated": time.strftime("%H:%M")}
        except Exception as e:
            logger.error(f"Market Data Error: {e}")
        return {"usd": "N/A", "updated": "--:--"}

    # --- 2. DEEP SCRAPER ---
    def scrape_article(self, url):
        """
        Follows redirects, finds the real High-Res Image, and extracts Article Text.
        """
        try:
            # 1. Resolve URL (Handle Google Redirects)
            resp = requests.get(url, headers=self._get_headers(), timeout=10)
            final_url = resp.url
            
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            # 2. Extract Image (OpenGraph)
            image_url = None
            meta_img = soup.find('meta', property='og:image')
            if meta_img: 
                image_url = urllib.parse.urljoin(final_url, meta_img['content'])

            # 3. Extract Text (Get paragraphs)
            # Remove clutter
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.extract()
            
            paragraphs = [p.get_text().strip() for p in soup.find_all('p')]
            # Keep only paragraphs with substantial text
            clean_text = " ".join([p for p in paragraphs if len(p) > 60])
            
            # Truncate to avoid token limits (approx 4000 chars)
            clean_text = clean_text[:4000]
            
            return final_url, image_url, clean_text

        except Exception as e:
            # logger.warning(f"Scrape failed for {url}: {e}")
            return url, None, ""

    # --- 3. ADVANCED AI ANALYST ---
    def analyze_with_ai(self, headline, full_text):
        """
        Uses Pollinations (OpenAI) to generate an Intelligence Report.
        """
        if not self.api_key: return None
        
        # If scrape failed and we have no text, use the headline
        context_text = full_text if len(full_text) > 100 else headline

        url = "https://gen.pollinations.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        system_prompt = (
            "You are an Elite Intelligence Analyst specializing in Iran."
            "Read the news text provided."
            "Output a strictly valid JSON object with the following fields:\n"
            "1. 'title_fa': Translate headline to professional Persian.\n"
            "2. 'summary': An array of 3 short, bullet-point strings in Persian summarizing the event.\n"
            "3. 'impact': A single sentence in Persian explaining the strategic impact on Iran.\n"
            "4. 'sentiment': A float from -1.0 (Critical/Negative for Iran) to 1.0 (Positive).\n"
            "5. 'tag': One category: [نظامی, هسته‌ای, اقتصادی, سیاسی, اجتماعی].\n"
            "Do not use markdown code blocks. Just the JSON."
        )

        user_content = f"HEADLINE: {headline}\n\nTEXT: {context_text}"

        payload = {
            "model": "openai",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.2
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                raw_content = response.json()['choices'][0]['message']['content']
                # Clean up formatting if AI adds ```json
                cleaned = raw_content.replace("```json", "").replace("```", "").strip()
                return json.loads(cleaned)
        except Exception as e:
            logger.error(f"AI Analysis Error: {e}")
        
        return None

    # --- 4. IMAGE GENERATOR ---
    def generate_ai_image(self, prompt):
        """Generates a fallback image if scraping fails."""
        try:
            safe_prompt = urllib.parse.quote(f"Editorial illustration, {prompt}, dark style, news context, highly detailed, 4k")
            return f"https://gen.pollinations.ai/image/{safe_prompt}?model=flux&width=800&height=600&nologo=true"
        except:
            return "https://placehold.co/800x600?text=News"

    # --- MAIN WORKER ---
    def process_item(self, entry):
        original_url = entry.get('url')
        
        # Quick check if we processed the raw google link
        if original_url in self.seen_urls: return None

        raw_title = entry.get('title', '').rsplit(' - ', 1)[0]
        date_str = entry.get('published date')

        # A. Scrape (Get Real URL + Text + Image)
        real_url, real_image, full_text = self.scrape_article(original_url)
        
        # Check if we processed the resolved URL
        if real_url in self.seen_urls: return None

        # B. Analyze (AI)
        ai_data = self.analyze_with_ai(raw_title, full_text)
        
        if not ai_data:
            # Fallback if API fails
            ai_data = {
                "title_fa": raw_title,
                "summary": ["متن کامل دریافت نشد", "تحلیل هوش مصنوعی در دسترس نیست"],
                "impact": "بدون تحلیل",
                "tag": "عمومی",
                "sentiment": 0
            }

        # C. Fallback Image
        if not real_image:
            real_image = self.generate_ai_image(raw_title)

        # D. Date Parsing (Crucial for Sorting)
        try:
            dt = parser.parse(date_str)
            timestamp = dt.timestamp()
        except:
            timestamp = time.time()

        return {
            "title_fa": ai_data.get('title_fa'),
            "title_en": raw_title,
            "summary": ai_data.get('summary'),
            "impact": ai_data.get('impact'),
            "tag": ai_data.get('tag'),
            "sentiment": ai_data.get('sentiment'),
            "source": entry.get('publisher', {}).get('title', 'Source'),
            "url": real_url,
            "image": real_image,
            "date": date_str,
            "timestamp": timestamp,
            "_original_url": original_url
        }

    def run(self):
        logger.info(">>> Radar System Started...")
        
        # 1. Update Market
        try:
            with open(CONFIG['FILES']['MARKET'], 'w', encoding='utf-8') as f:
                json.dump(self.fetch_market_rates(), f)
        except Exception: pass

        # 2. Get Google News
        gnews = GNews(language=CONFIG['LANGUAGE'], country=CONFIG['COUNTRY'], 
                      period=CONFIG['PERIOD'], max_results=CONFIG['MAX_RESULTS'])
        try:
            found_items = gnews.get_news(CONFIG['SEARCH_QUERY'])
        except Exception as e:
            logger.critical(f"GNews Failed: {e}")
            return

        logger.info(f">>> Found {len(found_items)} raw items. Processing...")

        new_items = []
        seen_updates = []

        # 3. Process in Parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG['MAX_WORKERS']) as executor:
            # Map futures to items
            future_map = {executor.submit(self.process_item, item): item for item in found_items}
            
            for future in concurrent.futures.as_completed(future_map):
                result = future.result()
                if result:
                    # Separate internal data from public data
                    orig_url = result.pop('_original_url')
                    seen_updates.append(orig_url) # Google URL
                    seen_updates.append(result['url']) # Real URL
                    
                    new_items.append(result)
                    logger.info(f" + [AI OK] {result['title_en'][:30]}...")

        # 4. Merge & Sort
        if new_items:
            try:
                with open(CONFIG['FILES']['NEWS'], 'r', encoding='utf-8') as f:
                    old_data = json.load(f)
            except: old_data = []

            # Combine new + old
            combined = new_items + old_data
            
            # Deduplicate by URL
            unique_dict = {item['url']: item for item in combined}
            unique_list = list(unique_dict.values())
            
            # Sort by Timestamp Descending (Newest First)
            unique_list.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
            
            # Trim to 60 items
            final_list = unique_list[:60]

            # Save
            with open(CONFIG['FILES']['NEWS'], 'w', encoding='utf-8') as f:
                json.dump(final_list, f, ensure_ascii=False, indent=4)
            
            self._save_seen(seen_updates)
            logger.info(f">>> Success. Saved {len(final_list)} items.")
        else:
            logger.info(">>> No new actionable intelligence found.")

if __name__ == "__main__":
    radar = IranNewsRadar()
    radar.run()
