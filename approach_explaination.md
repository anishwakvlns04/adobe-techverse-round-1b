# Approach Explanation – Adobe Techverse 2025: Challenge 1B

## Persona-Driven Document Intelligence System

An intelligent system was developed to process document collections and extract the most relevant sections based on a specified **persona** and their **job-to-be-done**. The objective was to surface only the most contextually useful content by ranking and analyzing sections from PDFs across different domains and layouts — all while staying within strict computational and runtime constraints.

## 🗂 Folder Structure

techverse_round_1b/
│
├── Collection_X/               # Input/output per test case  
│   ├── PDFs/                   # Source documents  
│   ├── challenge1b_input.json  # Persona, job, file list  
│   └── challenge1b_output.json # Final ranked results  
│
├── schema/                     # (Optional) output validation schema  
├── src/  
│   └── analyze_collections.py  # Main pipeline logic  
│
├── requirements.txt            # Python dependencies  
├── Dockerfile                  # CPU-only container setup  
└── approach_explaination.md    # Methodology write-up  




## 🧠 Solution Design

The core pipeline begins by reading the `challenge1b_input.json`, which provides the persona context, job requirements, and target PDF list. For each collection, the pipeline either utilizes **Challenge 1A outline JSONs** (if available) or falls back to heuristic-based segmentation using text features such as font size, boldness, spacing, and heading patterns.

Each document is broken down into sections and scored using a combination of:

- TF-IDF similarity to persona/job keywords,
- Content length and density,
- Custom heuristics for multilingual and semi-structured PDFs.

This produces a ranked list of sections that are contextually aligned with the persona’s goals.

## 📌 Section Ranking Strategy

To determine relevance:

- TF-IDF vectors are computed from both the persona/job description and section texts.
- A weighted scoring system evaluates the **keyword coverage**, **semantic overlap**, and **signal-to-noise ratio**.
- Each section is assigned an `importance_rank`, which is used to order the top N sections in the output JSON.

This ranking ensures only the most meaningful content is retained, avoiding overload and irrelevant data.

## ✂️ Subsection Analysis

Within each high-ranked section, refined sentence extraction is performed. Sentences are scored and filtered based on their relevance and diversity using TF-IDF and keyword overlap metrics.

The goal of this step is to provide precise, high-signal summaries of lengthy sections, tailored to the persona's focus area.

## 🧠 Generalization & Adaptability

The solution is designed to work across a variety of document types:

- Research papers
- Educational content
- Financial reports
- Travel guides, etc.

Fallback logic is robust enough to handle PDFs without structured outlines. Language-agnostic scoring ensures consistent results even in semi-formatted or non-English content.

## 🧾 Output Format

Each collection outputs a `challenge1b_output.json` file that includes:

- `metadata`: Input files, persona, job, timestamp
- `extracted_sections`: Ranked sections with titles and page numbers
- `subsection_analysis`: Refined and concise sentence-level insights

The output schema is consistent with the specifications provided by Adobe Techverse 2025.

## ⚙️ Compliance with Constraints

- ✅ Runs on **CPU only**
- ✅ Operates **fully offline**
- ✅ Model size < 1 GB (uses lightweight vectorization)
- ✅ Execution time < 60 seconds for 3–5 PDFs

## 🔍 Why Ranking?

Ranking enables the system to prioritize information critical to the persona’s goal. Rather than presenting all available content, it selects **the most relevant sections**, ensuring focus, clarity, and reduced cognitive load — especially important in multi-document environments.

## ✅ Use Case Scenarios

The system is capable of handling diverse use cases, including:

- 🧪 Researchers preparing literature reviews
- 📊 Analysts reviewing R&D investments and revenue trends
- 📚 Students identifying key topics for exam preparation
- ✈️ Travel planners organizing group itineraries

## 📦 Summary

A lightweight, persona-aware document intelligence pipeline was developed to extract, rank, and summarize content from PDF collections. The system is modular, generalizes across domains, and adheres strictly to runtime and system constraints — making it well-suited for the Challenge 1B requirements.
