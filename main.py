import os
import json
import time
import logging
import cloudscraper
import html
import re
import concurrent.futures
from datetime import datetime, timedelta, timezone # <--- UPDATED IMPORT
from bs4 import BeautifulSoup
from gnews import GNews
from fake_useragent import UserAgent
from dateutil import parser

# --- CONFIGURATION ---
CONFIG = {
    'SEARCH_QUERY': 'Iran AND (Israel OR USA OR nuclear OR conflict OR sanctions OR currency OR IRGC)',
    'LANGUAGE': 'en',
    'COUNTRY': 'US',
    'PERIOD': '1h',
    'MAX_RESULTS': 10,  # LIMITED TO 10 AS REQUESTED
    'FILES': {
        'NEWS': 'news.json',
        'MARKET': 'market.json'
    },
    'TELEGRAM': {
        'BOT_TOKEN': os.environ.get('TG_BOT_TOKEN'), 
        'CHANNEL_ID': os.environ.get('TG_CHANNEL_ID') 
    },
    'TIMEOUT': 20,
    'MAX_WORKERS': 4,
    'POLLINATIONS_KEY': os.environ.get('POLLINATIONS_API_KEY')
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

class IranNewsRadar:
    def __init__(self):
        self.scraper = cloudscraper.create_scraper(browser='chrome') 
        self.api_key = CONFIG['POLLINATIONS_KEY']
        self.existing_news = self._load_existing_news()
        
        self.seen_urls = {item.get('url') for item in self.existing_news if item.get('url')}
        self.seen_titles = {self._normalize_text(item.get('title_en', '')) for item in self.existing_news}

    def _get_headers(self):
        return {'User-Agent': UserAgent().random}

    def _normalize_text(self, text):
        if not text: return ""
        return re.sub(r'\W+', '', text).lower()

    def _load_existing_news(self):
        if not os.path.exists(CONFIG['FILES']['NEWS']): return []
        try:
            with open(CONFIG['FILES']['NEWS'], 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except: return []

    # --- MARKET DATA ---
    def fetch_market_rates(self):
        data = {"usd": "نامشخص", "oil": "نامشخص", "updated": "--:--"}
        try:
            resp = self.scraper.get("https://alanchand.com/en/currencies-price/usd", timeout=10)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                usd = soup.find('input', attrs={'data-curr': 'tmn'})
                if usd:
                    val = usd.get('data-price') or usd.get('value')
                    # Format as 65,000
                    if val: data["usd"] = f"{int(int(val.replace(',', '')) / 10):,}"
        except: pass

        try:
            resp = self.scraper.get("https://oilprice.com/oil-price-charts/46", timeout=10)
            soup = BeautifulSoup(resp.text, 'html.parser')
            oil = soup.select_one(".last_price")
            if oil: data["oil"] = oil.get_text().strip()
        except: pass

        data["updated"] = time.strftime("%H:%M")
        return data

    # --- TELEGRAM SENDER (STYLISH PERSIAN) ---
    def send_digest_to_telegram(self, items):
        token = CONFIG['TELEGRAM']['BOT_TOKEN']
        chat_id = CONFIG['TELEGRAM']['CHANNEL_ID']
        if not token or not chat_id: 
            logger.error("❌ Telegram Credentials Missing!")
            return

        # 1. Prepare Market Header
        try:
            with open(CONFIG['FILES']['MARKET'], 'r') as f: mkt = json.load(f)
            usd_price = mkt.get('usd', 'نامشخص')
            oil_price = mkt.get('oil', 'نامشخص')
            
            # Persian Market Text
            market_text = f"💵 <b>دلار:</b> {usd_price} تومان | 🛢 <b>نفت:</b> {oil_price} دلار"
        except: market_text = ""

        # --- CALCULATE IRAN TIME (UTC+3:30) ---
        utc_now = datetime.now(timezone.utc)
        iran_offset = timezone(timedelta(hours=3, minutes=30))
        current_time = utc_now.astimezone(iran_offset).strftime("%H:%M")
        
        # 2. Persian Header
        header = (
            f"📡 <b>رادار هوشمند اخبار ایران</b> | ⏱ {current_time}\n"
            f"{market_text}\n"
            f"➖➖➖➖➖➖➖➖➖➖\n\n"
        )
        
        # 3. Persian Footer
        footer = "\n📊 <a href='https://itsyebekhe.github.io/rasadai/'>مشاهده داشبورد کامل و آرشیو اخبار</a>"

        messages_to_send = []
        current_chunk = header

        for item in items:
            # Data Preparation
            title = str(item.get('title_fa', item.get('title_en')))
            source = str(item.get('source', 'Unknown'))
            url = str(item.get('url', ''))
            impact = str(item.get('impact', ''))
            urgency = item.get('urgency', 3)
            
            # Icons
            icon = "🔹"
            if urgency >= 8: icon = "🚨"
            elif urgency >= 6: icon = "⚠️"

            # Tag Handling
            raw_tag = item.get('tag', 'General')
            tag_str = str(raw_tag[0]) if isinstance(raw_tag, list) and raw_tag else str(raw_tag)

            # HTML Escaping
            safe_title = html.escape(title)
            safe_source = html.escape(source)
            safe_impact = html.escape(impact)
            safe_tag = html.escape(tag_str).replace(' ', '_')
            
            # Summary formatting
            summary_raw = item.get('summary', [])
            if isinstance(summary_raw, str): summary_raw = [summary_raw]
            # Use small bullet points
            safe_summary = "\n".join([f"▪️ {html.escape(str(s))}" for s in summary_raw])

            # 4. Construct Stylish Item HTML
            item_html = (
                f"{icon} <b><a href='{url}'>{safe_title}</a></b>\n"
                f"🗞 <i>منبع: {safe_source}</i>\n\n"
                f"📝 <b>خلاصه:</b>\n{safe_summary}\n\n"
                f"🎯 <b>تأثیر:</b> {safe_impact}\n\n"
                f"#{safe_tag}\n"
                f"〰️〰️〰️〰️〰️〰️〰️\n\n"
            )

            # Check Chunk Limit (Safety buffer)
            if len(current_chunk) + len(item_html) + len(footer) > 3900:
                messages_to_send.append(current_chunk + footer)
                current_chunk = header + item_html
            else:
                current_chunk += item_html

        if current_chunk != header:
            messages_to_send.append(current_chunk + footer)

        # Send
        api_url = f"https://api.telegram.org/bot{token}/sendMessage"
        for i, msg in enumerate(messages_to_send):
            try:
                resp = cloudscraper.create_scraper().post(api_url, json={
                    "chat_id": chat_id, 
                    "text": msg, 
                    "parse_mode": "HTML", 
                    "disable_web_page_preview": True
                })
                if resp.status_code == 200:
                    logger.info(f"✅ Message {i+1} Sent.")
                else:
                    logger.error(f"❌ Send Fail: {resp.text}")
            except Exception as e:
                logger.error(f"❌ Connection Fail: {e}")
            time.sleep(1.5)

    # --- SCRAPER & AI (Unchanged Logic) ---
    def scrape_article(self, url, fallback):
        try:
            if url.lower().endswith('.pdf'): return url, fallback
            resp = self.scraper.get(url, timeout=15, allow_redirects=True)
            soup = BeautifulSoup(resp.text, 'html.parser')
            for tag in soup(["script", "style", "nav", "footer", "header", "form", "iframe"]): tag.extract()
            paragraphs = [p.get_text().strip() for p in soup.find_all('p') if len(p.get_text()) > 50]
            text = " ".join(paragraphs)[:4500]
            if len(text) < 100: return resp.url, fallback
            return resp.url, text
        except: return url, fallback

    def analyze_with_ai(self, headline, full_text):
        if not self.api_key: return None
        context = full_text if len(full_text) > 100 else headline
        fallback = {"title_fa": headline, "summary": ["تحلیل در دسترس نیست"], "impact": "بررسی نشده", "tag": "News", "urgency": 3}

        try:
            resp = self.scraper.post(
                "https://gen.pollinations.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={
                    "model": "openai",
                    "messages": [{
                        "role": "system", 
                        "content": "You are a Persian News Analyst. Output valid JSON: {title_fa, summary[3 bullet points], impact(1 sentence), tag(1 word), urgency(1-10)}."
                    }, {
                        "role": "user", 
                        "content": f"HEADLINE: {headline}\nTEXT: {context}"
                    }],
                    "temperature": 0.1
                }, timeout=30
            )
            if resp.status_code == 200:
                clean = resp.json()['choices'][0]['message']['content'].replace('```json','').replace('```','').strip()
                return json.loads(clean)
        except: pass
        return fallback

    def process_item(self, entry):
        raw_title = entry.get('title', '').rsplit(' - ', 1)[0]
        norm_title = self._normalize_text(raw_title)
        
        if norm_title in self.seen_titles: return None
        
        orig_url = entry.get('url')
        if orig_url in self.seen_urls: return None

        snippet = entry.get('description', raw_title)
        real_url, text = self.scrape_article(orig_url, snippet)
        
        ai = self.analyze_with_ai(raw_title, text)
        if not ai: ai = {}

        try: ts = parser.parse(entry.get('published date')).timestamp()
        except: ts = time.time()

        return {
            "title_fa": ai.get('title_fa', raw_title),
            "title_en": raw_title,
            "summary": ai.get('summary', [snippet]),
            "impact": ai.get('impact', '...'),
            "tag": ai.get('tag', 'General'),
            "urgency": ai.get('urgency', 3),
            "source": entry.get('publisher', {}).get('title', 'Source'),
            "url": real_url,
            "date": entry.get('published date'),
            "timestamp": ts
        }

    def run(self):
        logger.info(">>> Radar Started...")
        with open(CONFIG['FILES']['MARKET'], 'w') as f: json.dump(self.fetch_market_rates(), f)

        try:
            results = GNews(language=CONFIG['LANGUAGE'], country=CONFIG['COUNTRY'], 
                           period=CONFIG['PERIOD'], max_results=CONFIG['MAX_RESULTS']).get_news(CONFIG['SEARCH_QUERY'])
        except Exception as e:
            logger.error(f"GNews Error: {e}")
            return

        new_items = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG['MAX_WORKERS']) as exc:
            futures = {exc.submit(self.process_item, i): i for i in results}
            for fut in concurrent.futures.as_completed(futures):
                res = fut.result()
                if res:
                    new_items.append(res)
                    self.seen_titles.add(self._normalize_text(res['title_en']))
                    logger.info(f" + OK: {res['title_en'][:20]}")

        if new_items:
            new_items.sort(key=lambda x: (x.get('urgency', 0), x.get('timestamp', 0)), reverse=True)
            self.send_digest_to_telegram(new_items)

            all_news = new_items + self.existing_news
            all_news.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
            with open(CONFIG['FILES']['NEWS'], 'w', encoding='utf-8') as f: 
                json.dump(all_news[:100], f, indent=4, ensure_ascii=False)
            logger.info(">>> Done.")
        else:
            logger.info(">>> No unique news.")

if __name__ == "__main__":
    IranNewsRadar().run()
