from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_and_save_text(input_path, output_path, chunk_size=1000, chunk_overlap=200):
    """
    Splits text into chunks and saves them to a file with markers.
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as file:
            text = file.read()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", " "]
        )
        chunks = splitter.split_text(text)

        with open(output_path, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks):
                f.write(f"--- Chunk {i + 1} ---\n{chunk}\n\n")
        print(f"Text split into {len(chunks)} chunks and saved to {output_path}")
        return chunks
    except Exception as e:
        print(f"Error chunking text: {e}")
        return []
