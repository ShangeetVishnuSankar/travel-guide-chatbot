"""
DATA COLLECTOR
==============
This module handles fetching and saving travel content from the web.

CONCEPTS YOU'LL LEARN:
- Web scraping basics with BeautifulSoup
- Document loading with LangChain's WebBaseLoader
- Why data quality matters in RAG systems
"""

import os
import requests

# --- CONFIGURATION ---
# Add Wikivoyage URLs for destinations you want your bot to know about.
# Start with 3-5 destinations. You can always add more later.

WIKIVOYAGE_URLS = [
    "https://en.wikivoyage.org/wiki/Bali",
    "https://en.wikivoyage.org/wiki/Tokyo",
    "https://en.wikivoyage.org/wiki/Paris",
    "https://en.wikivoyage.org/wiki/Dubai",
    "https://en.wikivoyage.org/wiki/New_York_City",
    "https://en.wikivoyage.org/wiki/Doha",
    "https://en.wikivoyage.org/wiki/Bangkok",
    "https://en.wikivoyage.org/wiki/Istanbul",
]

def fetch_wikivoyage_page(url: str) -> str:
    """
    Fetch a Wikivoyage page and extract the main text content.

    WHY USE THE API INSTEAD OF SCRAPING HTML?
    - The HTML page loads content dynamically, making scraping unreliable.
    - The MediaWiki API returns clean plain text directly — no parsing needed.
    - RAG works best with clean, relevant text. Garbage in = garbage out.
    """
    print(f"Fetching: {url}")

    # Extract the page title from the URL (e.g. "Bali" from ".../wiki/Bali")
    title = url.split("/wiki/")[-1]

    headers = {"User-Agent": "Mozilla/5.0 (compatible; travel-guide-bot/1.0)"}
    response = requests.get(
        "https://en.wikivoyage.org/w/api.php",
        headers=headers,
        params={
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "format": "json",
            "explaintext": "1",  # Return plain text, not HTML
        },
    )

    data = response.json()
    page = next(iter(data["query"]["pages"].values()))
    content = page.get("extract", "")

    if not content:
        print(f"  ⚠️  Could not find content for {url}")
        return ""

    return content

def save_destinations():
    """Download all destinations and save as text files in data/."""
    os.makedirs("data/web", exist_ok=True)

    for url in WIKIVOYAGE_URLS:
        # Extract destination name from URL
        destination = url.split("/wiki/")[-1].replace("_", " ")
        content = fetch_wikivoyage_page(url)

        if content:
            filepath = f"data/web/{destination.lower().replace(' ', '_')}.txt"
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"# Travel Guide: {destination}\n\n")
                f.write(content)
            print(f"  ✅ Saved: {filepath} ({len(content)} characters)")


if __name__ == "__main__":
    save_destinations()
    print("\n🎉 Data collection complete! Check the data/web/ folder.")