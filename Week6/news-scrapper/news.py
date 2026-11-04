import scrapy
import hashlib
import requests
from scrapy.http import Request
from scrapy.exceptions import NotConfigured
from scrapy.selector import Selector

class NewsSpider(scrapy.Spider):
    name = 'news'
    
    allowed_domains = [
        "reuters.com",
        "bloomberg.com",
        "forbes.com",
        "businessinsider.com",
        "marketwatch.com",
        "wsj.com",  # The Wall Street Journal
        "ft.com",   # Financial Times
        "cnbc.com", # CNBC
        "economist.com",  # The Economist
        "yahoo.com/finance",  # Yahoo Finance
        "investing.com",  # Investing.com
        "money.cnn.com",  # CNN Money
        "businessweek.com",  # Bloomberg Businessweek
        "moneymorning.com",  # Money Morning
        "nasdaq.com",  # Nasdaq
        "seekingalpha.com",  # Seeking Alpha
        "theguardian.com/uk/business",  # The Guardian (Business section)
        "republicworld.com/business",  # Republic World (India)
        "thefinancialbrand.com",  # The Financial Brand
        "morningstar.com"  # Morningstar
    ]

    custom_settings = {
        'FEED_FORMAT': 'json',
        'FEED_URI': 'output.json',
        'DOWNLOAD_DELAY': 2,  # Add delay between requests
        'CONCURRENT_REQUESTS_PER_DOMAIN': 8,  # Increase concurrency per domain
        'ROBOTSTXT_OBEY': False,
        'AUTOTHROTTLE_ENABLED': True,  # Auto-throttle for preventing overloading the server
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'  # Mimic a real browser
    }

    def start_requests(self):
        for domain in self.allowed_domains:
            url = f'https://{domain}'
            yield Request(url, callback=self.parse)

    def parse(self, response):
        try:
            # Extract all links that may potentially be articles (check if the href has "article" or similar keywords)
            article_links = response.css('a::attr(href)').getall()
            for link in article_links:
                # Ensure the link is a full URL (handle relative URLs)
                if link and ('article' in link or 'news' in link):  
                    if not link.startswith('http'):
                        link = response.urljoin(link)
                    yield response.follow(link, callback=self.parse_article)
        except Exception as e:
            self.logger.error(f"Error parsing {response.url}: {e}")

    def parse_article(self, response):
        try:
            title = self.extract_title(response)
            article_text = self.extract_article_text(response)
            if not article_text:
                self.logger.warning(f"No article text found for {response.url}")
                return

            cleaned_title = self.clean_text(title)
            article_id = hashlib.md5(cleaned_title.encode('utf-8')).hexdigest()

            author = self.extract_author(response)
            date = self.extract_date(response)
            summary = self.extract_summary(response)

            yield {
                'id': article_id,
                'url': response.url,
                'title': cleaned_title,
                'text': article_text,
                'author': author,
                'date': date,
                'summary': summary,
            }
        except Exception as e:
            self.logger.error(f"Error parsing article {response.url}: {e}")

    def extract_title(self, response):
        """ Extracts the title of the article, handling variations across sites. """
        title = response.css('h1::text').get()
        if not title:
            title = response.css('title::text').get()
        return title or "No Title Found"

    def extract_article_text(self, response):
        """ Extracts the main content of the article from <p> or similar tags. """
        article_text = ' '.join(response.css('div.article-body p::text, div.entry-content p::text, article p::text').getall()).strip()
        return article_text

    def extract_author(self, response):
        """ Extracts the author from multiple possible selectors. """
        author = response.css('span.byline__name::text, div.byline a::text, meta[name="author"]::attr(content)').get()
        return author or "No Author Found"

    def extract_date(self, response):
        """ Extracts the article date, handling variations in tag and attribute names. """
        date = response.css('time::attr(datetime), meta[property="article:published_time"]::attr(content)').get()
        if not date:
            date = response.css('meta[name="date"]::attr(content)').get()
        return date or "No Date Found"

    def extract_summary(self, response):
        """ Extracts a short summary or description of the article. """
        summary = response.css('div[itemprop="description"]::text, meta[name="description"]::attr(content)').get()
        return summary or "No Summary Found"

    def clean_text(self, text):
        """ Remove unwanted characters or whitespace. """
        return ' '.join(text.strip().split())