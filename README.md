# Adobe Techverse 2025 – Challenge 1B  

This project implements a smart document analysis system that extracts and ranks the most relevant sections from a collection of PDF documents. The system is guided by a given **persona** and their **job-to-be-done**, ensuring the output is aligned with user intent.

---

## 🚀 Key Features

- ✅ Persona- and job-aware section extraction
- 📑 Uses structured outlines from Round 1A when available
- 🔍 TF-IDF and keyword-based relevance scoring
- ✂️ Granular sub-section refinement using top sentence extraction
- ⚙️ Fully offline — no internet required during execution
- 🧠 Generalizable across domains (research, business, education, travel, etc.)
- 📄 JSON output matches `challenge1b_output.json` specification

---

## 🗂 Folder Structure

```

techverse\_round\_1b/
│
├── Collection\_X/                # Input/output for a document set
│   ├── PDFs/                    # Input PDFs
│   ├── challenge1b\_input.json   # Persona, job, input metadata
│   └── challenge1b\_output.json  # Final ranked result (generated)
│
├── schema/                      # (Optional) output format schema
├── src/
│   └── analyze\_collections.py   # Main processing script
│
├── requirements.txt             # Python package requirements
├── Dockerfile                   # For containerized execution
└── approach\_explaination.md     # Explanation of the methodology

````

---

## 🛠️ Setup & Execution

### ⚙️ Local Python Setup

> 💡 Python 3.10+ is required.

```bash

# (Optional) create virtual environment
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the processing script
python src/analyze_collections.py --root .
````

---

### 🐳 Docker Build & Run (Offline-Compatible)

```bash
# Build Docker image (inside techverse_round_1b)
docker build -t techverse-1b .

# Run the container (Linux/macOS)
docker run --rm -v $PWD:/app techverse-1b --root /app

# Run on Windows CMD
docker run --rm -v %cd%:/app techverse-1b --root /app
```

---

## 📤 Output Format

The system generates `challenge1b_output.json` in each `Collection_X/` folder. It contains:

* **Metadata**: Persona, job, input documents, and timestamp
* **Extracted Sections**: Section titles with page number and importance rank
* **Subsection Analysis**: Refined sentences providing focused insights

---

## 📌 Constraints Met

* ✅ CPU-only execution
* ✅ No internet access
* ✅ Model size < 1GB
* ✅ Execution time < 60 seconds (3–5 PDFs)

---

## 📄 Methodology

The complete methodology, including segmentation logic, ranking heuristics, TF-IDF scoring, and fallback strategies, is explained in detail in the `approach_explaination.md` file provided in the `src/` directory.

---

## ✅ Example Use Cases

* 🧪 Researcher preparing literature reviews
* 📊 Analyst comparing financial trends
* 📚 Student identifying exam topics
* ✈️ Travel planner building itinerary insights

---

## ℹ️ Note

If outline JSONs from Round 1A are present, they are used for accurate section segmentation. In their absence, regex- and heuristic-based segmentation ensures robustness across formats.

