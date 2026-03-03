import os
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from each page of a PDF file.
    """
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            full_text += f"\n--- Page {page_num + 1} ---\n{text}"
        doc.close()
        return full_text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def save_text_to_file(text, output_path):
    """
    Saves the extracted text to a file.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Text successfully saved to {output_path}")
    except Exception as e:
        print(f"Error saving text to file: {e}")

if __name__ == "__main__":
    # Example usage
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))

    medicalbook_file = os.path.join(project_root, 'data', 'raw', 'Medical_book.pdf')
    text_output = os.path.join(project_root, 'data', 'processed', 'medical_text.txt')

    extracted_text = extract_text_from_pdf(medicalbook_file)
    if extracted_text:
        save_text_to_file(extracted_text, text_output)
