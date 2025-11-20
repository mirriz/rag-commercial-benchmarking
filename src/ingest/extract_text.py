import os
from bs4 import BeautifulSoup

input_folder = 'data/10k-raw-html' 
output_folder = 'data/10k-txt'   


# Loop through raw TML files 
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".html"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".txt")

        print(f"{filename}")

        # Read content
        with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
            html_content = f.read()

        # Parse HTML
        soup = BeautifulSoup(html_content, "lxml")

        # Remove tags
        for tag in soup(["script", "style", "ix:header", "ix:hidden", "link", "meta", "title", "head"]):
            tag.decompose()

        # Extract text
        text = soup.get_text(separator="\n")

        # Clean whitespace
        clean_text = "\n".join(line.strip() for line in text.splitlines() if line.strip())

        # Write to text file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(clean_text)

        print(f"{output_path}")

print("\nAll files processed successfully!")
