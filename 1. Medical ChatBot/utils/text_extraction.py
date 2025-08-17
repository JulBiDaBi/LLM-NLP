import os 
import fitz  # PyMuPDF


HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(HOME)

# Define a function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Erreur lors de l'ouverture du fichier PDF : {e}")
        return ""

    full_text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        full_text += f"\n--- Page {page_num + 1} ---\n{text}"

    doc.close()
    return full_text

 
# Apply the function to our specic PDF Medical Book
medicalbook_file = os.path.join(HOME, 'Data', 'Medical_book.pdf')
extracted_text = extract_text_from_pdf(medicalbook_file)

# Sauvegarde dans un fichier texte
text_output_path = os.path.join(HOME, 'Data', 'processed')
text_output = os.path.join(text_output_path, 'medical_text.txt')


if not os.path.exists(text_output_path):
    os.makedirs(text_output_path)

with open(text_output, "w", encoding="utf-8") as f:
    f.write(extracted_text)
 
print(
    f"\n----------------------------------------------------\n"
    f"Text successfully extracted from {medicalbook_file} and saved to {text_output}"
    "\n----------------------------------------------------\n"
)