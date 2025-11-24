import os
import re
from bs4 import BeautifulSoup

# --- CONFIGURATION ---
HTML_INPUT_FOLDER = 'data/10k-raw-html' 
TXT_OUTPUT_FOLDER = 'data/10ks-raw'   

def clean_financial_string(text):
    """
    Cleans specific financial report artifacts.
    """

    text = text.replace('\u200b', '')
    
    
    # 3. Collapse excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Converts unicode chars to readable text
    text = text.replace('☒', '[x]').replace('☐', '[ ]')
    

    return text.strip()


def html_to_raw_text():
    """
    STEP 1: Reads HTML files, strips tags, and saves raw text.
    (You have likely already run this).
    """
    print("--- STEP 1: Converting HTML to Raw Text ---")
    
    if not os.path.exists(HTML_INPUT_FOLDER):
        print(f"Directory not found: {HTML_INPUT_FOLDER}")
        return

    os.makedirs(TXT_OUTPUT_FOLDER, exist_ok=True)

    for filename in os.listdir(HTML_INPUT_FOLDER):
        if filename.lower().endswith(".html"):
            input_path = os.path.join(HTML_INPUT_FOLDER, filename)
            output_path = os.path.join(TXT_OUTPUT_FOLDER, os.path.splitext(filename)[0] + ".txt")

            print(f"Processing HTML: {filename}")
            try:
                with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
                    html_content = f.read()

                soup = BeautifulSoup(html_content, "lxml")
                
                # Remove script/style tags
                for tag in soup(["script", "style", "ix:header", "ix:hidden", "link", "meta", "title", "head"]):
                    tag.decompose()

                text = soup.get_text(separator="\n")
                
                # Basic whitespace trim
                clean_text = "\n".join(line.strip() for line in text.splitlines() if line.strip())

                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(clean_text)
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")



def clean_existing_text_files():
    """
    Reads text files and applies cleaning.
    """
    print("\n--- Scrubbing Existing Text Files ---")
    
    if not os.path.exists(TXT_OUTPUT_FOLDER):
        print(f"Directory not found: {TXT_OUTPUT_FOLDER}")
        return

    files = [f for f in os.listdir(TXT_OUTPUT_FOLDER) if f.lower().endswith(".txt")]
    print(f"Found {len(files)} text files to clean.")

    for filename in files:
        file_path = os.path.join(TXT_OUTPUT_FOLDER, filename)
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Apply the cleaning
            cleaned_content = clean_financial_string(content)
            
            # Overwrite the file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(cleaned_content)

            
        except Exception as e:
            print(f"Error scrubbing {filename}: {e}")
            
    print("Text scrubbing complete.")

if __name__ == "__main__":
    
    # html_to_raw_text()

    clean_existing_text_files()