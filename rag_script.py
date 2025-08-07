import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def load_and_chunk_docs(folder_path, chunk_size=300):
    chunks = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                words = text.split()
                for i in range(0, len(words), chunk_size):
                    chunk = ' '.join(words[i:i+chunk_size])
                    chunks.append(chunk)
                    print(f"✅ Created chunk from {filename} with {len(words[i:i+chunk_size])} words.")
    return chunks

chunks = load_and_chunk_docs("docs")
print("\n🔶 Total chunks created:", len(chunks))


print("\n🔷 Generating embeddings...")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)
print(f"✅ Embeddings created. Shape: {len(embeddings)} embeddings of dimension {len(embeddings[0])}")


embedding_matrix = np.array(embeddings).astype("float32")
dimension = embedding_matrix.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embedding_matrix)

print("\n✅ FAISS index created and embeddings added.")
print("Total vectors in index:", index.ntotal)

while True:
    query = input("\n❓ Enter your question (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        break

    query_embedding = model.encode([query]).astype("float32")

    
    top_k = 1  # number of results to return
    distances, indices = index.search(query_embedding, top_k)

    
    answer = chunks[indices[0][0]]

    print("\n💡 Most relevant chunk:")
    print(answer)

