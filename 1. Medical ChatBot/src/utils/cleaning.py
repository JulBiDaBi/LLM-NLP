import re

def clean_text(input_path, output_path):
    """
    Cleans the medical text by removing headers, page markers, and extra whitespace.
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as file:
            text = file.read()

        text = re.sub(r'\n--- Page \d+ ---\n', '\n', text)
        text = re.sub(r"GALE ENCLYCLOPEDIA OF MEDICINE 2\s*\w*\n*", '', text, flags=re.IGNORECASE)
        text = re.sub(r"Copyright © \d{4}.*\n", "", text)
        text = re.sub(r"ISBN [\d\-\(\) ]+\n*", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"\n?\s*\d+\s*\n", "\n", text)
        text = "\n".join(line.strip() for line in text.splitlines())
        text = re.sub(r"\n{3,}", "\n\n", text)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Text cleaned and saved to {output_path}")
    except Exception as e:
        print(f"Error cleaning text: {e}")
