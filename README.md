# Multimodal Fashion Search

A robust multimodal machine learning project designed to bridge text and visual data for advanced fashion item retrieval. This system solves the problem of cross-modal search—allowing users to search through a massive catalog of apparel using natural language descriptions or reference images.

---

## 🚀 Project Overview & Problem Solved
Traditional search engines rely strictly on metadata or tags, which often fail when users look for specific visual styles, textures, or combinations described in plain text. This project implements a **Multimodal Embedding Space** using deep learning to map both product images and textual descriptions into a shared vector space. 

* **Text-to-Image Search:** Type a description (e.g., *"casual blue summer dress"*) to instantly retrieve matching fashion items.
* **Image-to-Image Search:** Upload a reference image to find visually similar styles in the catalog.
* **Vector Indexing:** Built with FAISS (Facebook AI Similarity Search) for ultra-fast, high-dimensional nearest-neighbor retrieval.

---

## 🛠️ Tech Stack & Architecture
* **Core ML:** PyTorch / TensorFlow, Feature Embeddings (`.npy`, `.index`)
* **Vector Search:** FAISS Index
* **Backend Framework:** FastAPI (Python)
* **Frontend UI:** HTML5, CSS3, JavaScript (Responsive layout)
* **Containerization:** Docker
* **Version Control:** Git & GitHub

---

## 📂 Project Structure
```text
Mid_Project/
│
├── main.py                # FastAPI backend server and search routes
├── index.html             # Web user interface for searching
├── Dockerfile             # Container configuration for deployment
├── requirements.txt       # Python dependencies
├── .gitignore             # Git exclusion rules for heavy weights/virtual envs
└── static/
    └── images/            # Product dataset images
Run the app locally with Docker:

```bash
docker build -t fashion-app .
docker run -p 8000:8000 fashion-app
```

Open http://localhost:8000 in a browser.

Product images are served from `static/images/`. Copy the dataset image files into
that directory (for example, `static/images/15970.jpg`) so they can be displayed
by the search results.
![Fashion Search UI](screenshots/screencapture-localhost-8000-2026-08-24-105229.png)
