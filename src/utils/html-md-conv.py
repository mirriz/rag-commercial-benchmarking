import os
from bs4 import BeautifulSoup
from markdownify import markdownify as md

# --- CONFIG ---
INPUT_DIR = "10k/"   # Place your raw HTML files here
OUTPUT_DIR = "10k-markdown/"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def clean_and_convert(html_content):
    """
    Parses HTML, removes noise, and converts tables to Markdown.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 1. Remove script and style tags (Noise)
    for script in soup(["script", "style", "head", "meta", "noscript"]):
        script.decompose()
        
    # 2. Convert to Markdown
    # 'tables' option ensures <table> tags become Markdown tables
    markdown_text = md(str(soup), heading_style="ATX", strip=['a', 'img'])
    
    # 3. Post-processing cleanup (optional)
    # Remove excessive newlines often left by conversion
    lines = markdown_text.split('\n')
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    return '\n\n'.join(cleaned_lines)

def process_html_files():
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.html', '.htm'))]
    print(f"Found {len(files)} HTML files. Converting to Markdown...")
    
    for i, filename in enumerate(files):
        input_path = os.path.join(INPUT_DIR, filename)
        output_filename = filename.rsplit('.', 1)[0] + ".md"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        if os.path.exists(output_path):
            print(f"Skipping {filename} (already exists)")
            continue
            
        print(f"[{i+1}/{len(files)}] Processing {filename}...")
        
        try:
            with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()
                
            markdown_content = clean_and_convert(html_content)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"\nSuccess! Markdown files are in {OUTPUT_DIR}")

if __name__ == "__main__":
    process_html_files()