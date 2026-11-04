# News Scraper - Multi-Source Financial News Spider

A Scrapy-based web scraper for collecting news articles from major financial and business news websites.

## 📋 Overview

This spider crawls and extracts articles from 20+ financial news sources including Reuters, Bloomberg, Forbes, WSJ, CNBC, and more. It automatically extracts article metadata including title, content, author, publication date, and summary.

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Install Scrapy:
```bash
pip install scrapy
```

2. Create a new Scrapy project:
```bash
scrapy startproject news
```

3. Navigate to the spiders directory:
```bash
cd news/spiders
```

4. Copy the spider file:
```bash
cp /path/to/news.py .
```

5. Return to the project root:
```bash
cd ..
```

### Running the Spider

**Basic usage:**
```bash
scrapy crawl news
```

**Save to JSON file:**
```bash
scrapy crawl news -o articles.json
```

**Save to CSV file:**
```bash
scrapy crawl news -o articles.csv
```

**Save to JSONL (JSON Lines) format:**
```bash
scrapy crawl news -o articles.jsonl
```

## 📊 Extracted Data Fields

Each article yields the following fields:

| Field | Description | Example |
|-------|-------------|---------|
| `id` | MD5 hash of the cleaned title | `a1b2c3d4e5f6...` |
| `url` | Full URL of the article | `https://reuters.com/article/...` |
| `title` | Article headline | `"Fed Raises Interest Rates..."` |
| `text` | Full article body text | `"The Federal Reserve..."` |
| `author` | Article author(s) | `"John Smith"` |
| `date` | Publication date/time | `"2026-04-21T10:30:00Z"` |
| `summary` | Article summary/description | `"The Fed announces..."` |

## ⚙️ Configuration

The spider includes the following custom settings:

```python
FEED_FORMAT = 'json'                    # Default output format
FEED_URI = 'output.json'               # Default output file
DOWNLOAD_DELAY = 2                     # 2 seconds between requests
CONCURRENT_REQUESTS_PER_DOMAIN = 8    # Max 8 concurrent requests per domain
ROBOTSTXT_OBEY = False                # Ignore robots.txt (use responsibly!)
AUTOTHROTTLE_ENABLED = True           # Auto-adjust crawl speed
USER_AGENT = 'Mozilla/5.0...'         # Mimic real browser
```

### Customizing Settings

**Modify download delay:**
```bash
scrapy crawl news -s DOWNLOAD_DELAY=5
```

**Change concurrent requests:**
```bash
scrapy crawl news -s CONCURRENT_REQUESTS_PER_DOMAIN=4
```

**Respect robots.txt:**
```bash
scrapy crawl news -s ROBOTSTXT_OBEY=True
```

## 📁 Project Structure

```
news-scrapper/
├── README.md           # This file
├── news.py            # Spider implementation
└── news/              # Scrapy project (after setup)
    ├── scrapy.cfg
    ├── news/
    │   ├── __init__.py
    │   ├── items.py
    │   ├── middlewares.py
    │   ├── pipelines.py
    │   ├── settings.py
    │   └── spiders/
    │       ├── __init__.py
    │       └── news.py    # Copy news.py here
    └── output.json    # Generated output
```

## 🔧 Advanced Usage

### Limiting Crawl Depth

```bash
scrapy crawl news -s DEPTH_LIMIT=2
```

### Setting Maximum Pages

```bash
scrapy crawl news -s CLOSESPIDER_PAGECOUNT=100
```

### Filtering by Domain

Edit `news.py` and modify the `allowed_domains` list to include only desired sources.

### Custom Output Processing

Create a pipeline in `news/pipelines.py`:

```python
class NewsScraperPipeline:
    def process_item(self, item, spider):
        # Clean or transform data
        item['title'] = item['title'].upper()
        return item
```

Enable in `settings.py`:
```python
ITEM_PIPELINES = {
    'news.pipelines.NewsScraperPipeline': 300,
}
```

## 🛠️ Troubleshooting

### Common Issues

**1. No articles found**
- Many sites use dynamic JavaScript rendering
- Try using Scrapy-Splash or Selenium for JavaScript-heavy sites
- Check if the site has changed its HTML structure

**2. Getting blocked**
- Increase `DOWNLOAD_DELAY` to 3-5 seconds
- Reduce `CONCURRENT_REQUESTS_PER_DOMAIN`
- Use rotating user agents or proxies

**3. Incomplete article text**
- Article selectors may need updating for specific sites
- Customize `extract_article_text()` method for problematic domains

**4. Rate limiting / 429 errors**
```bash
scrapy crawl news -s DOWNLOAD_DELAY=5 -s CONCURRENT_REQUESTS_PER_DOMAIN=2
```

### Debug Mode

Run with verbose logging:
```bash
scrapy crawl news -L DEBUG
```

## ⚠️ Legal & Ethical Considerations

- **Respect robots.txt**: Many sites block scrapers. Use responsibly.
- **Rate limiting**: Don't overwhelm servers with requests.
- **Terms of Service**: Review each site's ToS before scraping.
- **Copyright**: Scraped content may be copyrighted.
- **Personal use**: This tool is for educational/research purposes.

## 📝 Example Output

```json
{
  "id": "a1b2c3d4e5f6789",
  "url": "https://reuters.com/markets/us/fed-raises-rates-2026-04-21/",
  "title": "Fed Raises Interest Rates by 25 Basis Points",
  "text": "The Federal Reserve raised interest rates by 25 basis points...",
  "author": "Jane Doe",
  "date": "2026-04-21T14:30:00Z",
  "summary": "The Fed announces a quarter-point rate hike amid inflation concerns."
}
```

## 🔄 Monitoring Progress

```bash
# With statistics
scrapy crawl news -s LOG_LEVEL=INFO

# Real-time stats
scrapy crawl news -s STATSMAILER_RCPTS=['your@email.com']
```

## 📚 Additional Resources

- [Scrapy Documentation](https://docs.scrapy.org/)
- [Scrapy Tutorial](https://docs.scrapy.org/en/latest/intro/tutorial.html)
- [XPath & CSS Selectors](https://docs.scrapy.org/en/latest/topics/selectors.html)

## 🤝 Contributing

To add new news sources:

1. Add domain to `allowed_domains` list
2. Test selectors for that site
3. Customize extraction methods if needed

## 📄 License

Educational use only. Respect website terms of service and copyright laws.

---

**Last Updated**: April 2026

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Install Scrapy:
```bash
pip install scrapy
```

2. Create a new Scrapy project:
```bash
scrapy startproject news
```

3. Navigate to the spiders directory:
```bash
cd news/spiders
```

4. Copy the spider file:
```bash
cp /path/to/news.py .
```

5. Return to the project root:
```bash
cd ..
```

### Running the Spider

**Basic usage:**
```bash
scrapy crawl news
```

**Save to JSON file:**
```bash
scrapy crawl news -o articles.json
```

**Save to CSV file:**
```bash
scrapy crawl news -o articles.csv
```

**Save to JSONL (JSON Lines) format:**
```bash
scrapy crawl news -o articles.jsonl
```

## 📊 Extracted Data Fields

Each article yields the following fields:

| Field | Description | Example |
|-------|-------------|---------|
| `id` | MD5 hash of the cleaned title | `a1b2c3d4e5f6...` |
| `url` | Full URL of the article | `https://reuters.com/article/...` |
| `title` | Article headline | `"Fed Raises Interest Rates..."` |
| `text` | Full article body text | `"The Federal Reserve..."` |
| `author` | Article author(s) | `"John Smith"` |
| `date` | Publication date/time | `"2026-04-21T10:30:00Z"` |
| `summary` | Article summary/description | `"The Fed announces..."` |

## ⚙️ Configuration

The spider includes the following custom settings:

```python
FEED_FORMAT = 'json'                    # Default output format
FEED_URI = 'output.json'               # Default output file
DOWNLOAD_DELAY = 2                     # 2 seconds between requests
CONCURRENT_REQUESTS_PER_DOMAIN = 8    # Max 8 concurrent requests per domain
ROBOTSTXT_OBEY = False                # Ignore robots.txt (use responsibly!)
AUTOTHROTTLE_ENABLED = True           # Auto-adjust crawl speed
USER_AGENT = 'Mozilla/5.0...'         # Mimic real browser
```

### Customizing Settings

**Modify download delay:**
```bash
scrapy crawl news -s DOWNLOAD_DELAY=5
```

**Change concurrent requests:**
```bash
scrapy crawl news -s CONCURRENT_REQUESTS_PER_DOMAIN=4
```

**Respect robots.txt:**
```bash
scrapy crawl news -s ROBOTSTXT_OBEY=True
```

## 📁 Project Structure

```
news-scrapper/
├── README.md           # This file
├── news.py            # Spider implementation
└── news/              # Scrapy project (after setup)
    ├── scrapy.cfg
    ├── news/
    │   ├── __init__.py
    │   ├── items.py
    │   ├── middlewares.py
    │   ├── pipelines.py
    │   ├── settings.py
    │   └── spiders/
    │       ├── __init__.py
    │       └── news.py    # Copy news.py here
    └── output.json    # Generated output
```

## 🔧 Advanced Usage

### Limiting Crawl Depth

```bash
scrapy crawl news -s DEPTH_LIMIT=2
```

### Setting Maximum Pages

```bash
scrapy crawl news -s CLOSESPIDER_PAGECOUNT=100
```

### Filtering by Domain

Edit `news.py` and modify the `allowed_domains` list to include only desired sources.

### Custom Output Processing

Create a pipeline in `news/pipelines.py`:

```python
class NewsScraperPipeline:
    def process_item(self, item, spider):
        # Clean or transform data
        item['title'] = item['title'].upper()
        return item
```

Enable in `settings.py`:
```python
ITEM_PIPELINES = {
    'news.pipelines.NewsScraperPipeline': 300,
}
```

## 🛠️ Troubleshooting

### Common Issues

**1. No articles found**
- Many sites use dynamic JavaScript rendering
- Try using Scrapy-Splash or Selenium for JavaScript-heavy sites
- Check if the site has changed its HTML structure

**2. Getting blocked**
- Increase `DOWNLOAD_DELAY` to 3-5 seconds
- Reduce `CONCURRENT_REQUESTS_PER_DOMAIN`
- Use rotating user agents or proxies

**3. Incomplete article text**
- Article selectors may need updating for specific sites
- Customize `extract_article_text()` method for problematic domains

**4. Rate limiting / 429 errors**
```bash
scrapy crawl news -s DOWNLOAD_DELAY=5 -s CONCURRENT_REQUESTS_PER_DOMAIN=2
```

### Debug Mode

Run with verbose logging:
```bash
scrapy crawl news -L DEBUG
```

## ⚠️ Legal & Ethical Considerations

- **Respect robots.txt**: Many sites block scrapers. Use responsibly.
- **Rate limiting**: Don't overwhelm servers with requests.
- **Terms of Service**: Review each site's ToS before scraping.
- **Copyright**: Scraped content may be copyrighted.
- **Personal use**: This tool is for educational/research purposes.

## 📝 Example Output

```json
{
  "id": "a1b2c3d4e5f6789",
  "url": "https://reuters.com/markets/us/fed-raises-rates-2026-04-21/",
  "title": "Fed Raises Interest Rates by 25 Basis Points",
  "text": "The Federal Reserve raised interest rates by 25 basis points...",
  "author": "Jane Doe",
  "date": "2026-04-21T14:30:00Z",
  "summary": "The Fed announces a quarter-point rate hike amid inflation concerns."
}
```

## 🔄 Monitoring Progress

```bash
# With statistics
scrapy crawl news -s LOG_LEVEL=INFO

# Real-time stats
scrapy crawl news -s STATSMAILER_RCPTS=['your@email.com']
```

## 📚 Additional Resources

- [Scrapy Documentation](https://docs.scrapy.org/)
- [Scrapy Tutorial](https://docs.scrapy.org/en/latest/intro/tutorial.html)
- [XPath & CSS Selectors](https://docs.scrapy.org/en/latest/topics/selectors.html)

