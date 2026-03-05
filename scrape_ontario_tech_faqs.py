"""
Ontario Tech University - Multi-Section FAQ Scraper
=====================================================
Scrapes FAQs from:
  - Undergraduate Admissions site (admissions.ontariotechu.ca)
  - Office of the Registrar site (registrar.ontariotechu.ca)

Output: ontario_tech_faqs_all.csv

Requirements:
    pip install requests beautifulsoup4
"""

import csv
import time
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# All FAQ pages to scrape
# ---------------------------------------------------------------------------
FAQ_PAGES = [
    # --- Admissions site ---
    {
        "url": "https://admissions.ontariotechu.ca/faqs/index.php",
        "section": "General Admissions",
    },
    {
        "url": "https://admissions.ontariotechu.ca/faqs/applicant-portal.php",
        "section": "Applicant Portal",
    },

    # --- Registrar site ---
    {
        "url": "https://registrar.ontariotechu.ca/faqs/admissions.php",
        "section": "Admissions",
    },
    {
        "url": "https://registrar.ontariotechu.ca/faqs/ancillary-fees.php",
        "section": "Ancillary Fees",
    },
    {
        "url": "https://registrar.ontariotechu.ca/faqs/awards-bursaries-scholarships.php",
        "section": "Awards, Bursaries and Scholarships",
    },
    {
        "url": "https://registrar.ontariotechu.ca/faqs/fees-and-payment.php",
        "section": "Fees and Payment",
    },
    {
        "url": "https://registrar.ontariotechu.ca/faqs/financial-aid-and-osap.php",
        "section": "Financial Aid and OSAP",
    },
    {
        "url": "https://registrar.ontariotechu.ca/faqs/graduating-students-and-convocation.php",
        "section": "Graduating Students and Convocation",
    },
    {
        "url": "https://registrar.ontariotechu.ca/faqs/faqs.php",
        "section": "Grad Finance FAQs and Contacts",
    },
    {
        "url": "https://registrar.ontariotechu.ca/faqs/registering-for-courses.php",
        "section": "Registering for Courses",
    },
    {
        "url": "https://registrar.ontariotechu.ca/faqs/registration-and-student-records.php",
        "section": "Registration and Student Records",
    },
    {
        "url": "https://registrar.ontariotechu.ca/faqs/other-services-and-departments.php",
        "section": "Other Services and Departments",
    },
]

OUTPUT_FILE = "ontario_tech_faqs_all.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


# ---------------------------------------------------------------------------
# Scraping logic
# ---------------------------------------------------------------------------

def scrape_page(url: str, section: str) -> list[dict]:
    """Fetch one FAQ page and return a list of {Section, Question, Answer} dicts."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"  ⚠️  Could not fetch {url}: {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")

    # Try to isolate the main content area to avoid nav/footer noise
    main = (
        soup.find("main")
        or soup.find("div", {"id": "main-content"})
        or soup.find("div", {"class": "content"})
        or soup
    )

    faqs = []
    seen = set()  # deduplicate within the same page

    for li in main.find_all("li"):
        lines = [
            line.strip()
            for line in li.get_text(separator="\n").splitlines()
            if line.strip()
        ]

        if not lines:
            continue

        question = lines[0]

        # Keep only lines that look like FAQ questions
        if "?" not in question or len(question) < 15:
            continue

        # Everything after the first line is the answer
        answer = " ".join(lines[1:]).strip()

        if not answer:
            continue

        # Skip duplicates on the same page
        key = question.lower()
        if key in seen:
            continue
        seen.add(key)

        faqs.append({
            "Section": section,
            "Question": question,
            "Answer": answer,
        })

    return faqs


def scrape_all_pages(pages: list[dict]) -> list[dict]:
    """Iterate over all pages and collect every FAQ."""
    all_faqs = []

    for page in pages:
        print(f"Scraping [{page['section']}] → {page['url']}")
        faqs = scrape_page(page["url"], page["section"])
        print(f" Found {len(faqs)} FAQs")
        all_faqs.extend(faqs)
        time.sleep(0.5)  # be polite to the server

    return all_faqs


def save_to_csv(faqs: list[dict], output_file: str):
    """Write all FAQs to a CSV file."""
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Section", "Question", "Answer"])
        writer.writeheader()
        writer.writerows(faqs)
    print(f"\n Saved {len(faqs)} total FAQs to '{output_file}'")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("🔍 Ontario Tech FAQ Scraper — Multi-Section\n")

    all_faqs = scrape_all_pages(FAQ_PAGES)

    if all_faqs:
        save_to_csv(all_faqs, OUTPUT_FILE)

        # Preview
        print("\n--- Preview (first 3 results) ---")
        for i, faq in enumerate(all_faqs[:3], 1):
            print(f"\n[{i}] Section : {faq['Section']}")
            print(f"     Question: {faq['Question']}")
            print(f"     Answer  : {faq['Answer'][:120]}...")
    else:
        print(" No FAQs found. Check the URLs or page structure.")
