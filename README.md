# 🪄 Building a Knowledge Graph for the Harry Potter Series

This project builds an interactive **Knowledge Graph** of the Harry Potter universe, mapping out characters, their communities, and their relationships based on the **book series** and **wiki fandom** data.

---

## 📌 Overview

The goal of this project was to understand and visualize **character relationships** in the Harry Potter books by:
- Extracting characters and communities from structured and unstructured sources.
- Building a **co-occurrence network** to model relationships.
- Analyzing the network with **graph theory** and **centrality measures**.
- Creating both **static** and **interactive** network visualizations.

---

## 🛠️ Methodology

### 1. **Data Collection**
- **Web Scraping**: Used **Selenium** to scrape character data from the [Harry Potter Wiki Fandom](https://harrypotter.fandom.com/).
- **Book Parsing**: Used the **text versions of the books** to extract occurrences of character names.
- Combined both sources to create a comprehensive list of named entities.

### 2. **Named Entity Recognition (NER)**
- Applied **SpaCy** for NER to automatically detect **people** and other named entities.
- Mapped different name variations to the same character (e.g., “Harry” and “Harry Potter”).

### 3. **Relationship Network**
- Defined **co-occurrence** as two characters appearing in the same sentence.
- Assigned weights to each pair of chaacters based on their number of co-occurences.
- Represented the results as a dataframe for network creation.

### 4. **Graph Construction**
- Built the network using **NetworkX** for analysis.
- Created an interactive version with **PyVis** for web-based exploration.

### 5. **Graph Analysis**
- Measured:
  - **Degree Centrality**
  - **Betweenness Centrality**
  - **Closeness Centrality**  
  to identify key characters.
- Grouped characters into **communities** using modularity-based clustering.

### 6. **Visualization**
- **Communities**: Highlighted groups of closely connected characters.
- **Static Graphs**: Used NetworkX for detailed static plots.
- **Interactive Graph**: PyVis network for interactive exploration.

---

## 📊 Results & Visuals

### Communities in the Harry Potter Universe
![Communities](https://github.com/AmrinKareem/Knowledge_Graph_Harry_Potter/blob/main/community.png)

---

### Interactive Character Network
![Characters](https://github.com/AmrinKareem/Knowledge_Graph_Harry_Potter/blob/main/pyvis.png)

---

### Most Important Characters (Centrality Analysis)
![Important Characters](https://github.com/AmrinKareem/Knowledge_Graph_Harry_Potter/blob/main/output.png)

---

## 📈 Key Insights
- **Harry Potter**, **Albus Dumbledore**, and **Hermione Granger** consistently ranked highest across centrality measures.
- Community detection revealed **distinct social groups** (e.g., Hogwarts Houses, Order of the Phoenix, Death Eaters).
- Co-occurrence networks highlight **hidden relationships** not explicitly stated in the books.

---

## 📦 Tech Stack
- **Python**
- **Selenium** – web scraping
- **SpaCy** – NER
- **Pandas / NumPy** – data manipulation
- **NetworkX / PyVis** – graph creation & visualization
- **Matplotlib / Seaborn** – plotting

---

## 🚀 Future Work
- Incorporate **sentiment analysis** for each character relationship.
- Build a **time-evolving graph** to show how relationships change over the series.

---

## Credits
 - Inspired by Thu Vu’s Witcher Knowledge Graph project – Thu Vu Data Analytics
 - J.K. Rowling for the Harry Potter series.
 - Harry Potter Wiki Fandom for structured character information.

## 📜 How to Run
```bash
git clone https://github.com/AmrinKareem/Knowledge_Graph_Harry_Potter.git
cd Knowledge_Graph_Harry_Potter
pip install -r requirements.txt
python main.py
