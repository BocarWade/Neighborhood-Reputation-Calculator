# Neighborhood Sentiment Analysis

## Overview  
This repository provides a **Neighborhood Sentiment Analysis** tool that allows users to calculate and analyze the reputation of neighborhoods based on sentiment extracted from text-based data (e.g., newspaper articles). The goal is to empower others to evaluate the sentiment and perception of specific neighborhoods in their cities by leveraging pre-trained sentiment models.  

The project uses **VADER (Valence Aware Dictionary and Sentiment Reasoner)** to analyze the sentiment of individual articles, then aggregates these scores to provide insights into neighborhood reputation trends over time.  

---

## Features  
- **Sentiment Analysis**: Evaluates the sentiment of articles using VADER.  
- **Neighborhood-Level Insights**: Aggregates scores to assess the sentiment for specific neighborhoods.  
- **Historical Sentiment Trends**: Provides tools to analyze how sentiment changes over time (e.g., per year).  
- **Statistics and Reporting**: Includes functions for calculating averages, counts, and other statistics to help users understand their data.

---

## How It Works  
The workflow consists of:  
1. **Reading Input Files**: Accepts CSV files containing articles, each with text and date columns.  
2. **Sentiment Scoring**: Uses VADER to analyze the sentiment of each article.  
3. **Aggregation**: Computes average sentiment scores for each year and neighborhood.  
4. **Exporting Results**: Saves neighborhood-level sentiment data to pickle files for future analysis.
5. **Visualization**: Provides tools for graphing yearly sentiment trends.

---

## Folder Structure  
```
📂 Neighborhood-Sentiment-Analysis/
│
├── data/                    # Directory containing input CSV files
│   ├── Midtown.csv
│   └── Englewood.csv
├── Sentiment_Scores_Final/  # Directory for storing sentiment score pickle files
├── Date_Scores_Final/       # Directory for storing year-based sentiment pickle files
├── yearly_counts.csv        # CSV output of yearly sentiment counts per neighborhood
├── neighborhood_analysis.py # Main script containing all functions
└── README.md                # Project overview and instructions (this file)
```

---

## Dependencies  
Make sure you have the following libraries installed:  
- Python 3.x  
- pandas  
- numpy  
- nltk  
- tensorflow  
- vaderSentiment  
- matplotlib  
- BeautifulSoup (bs4)  
- lxml  

You can install them using:  
```bash
pip install -r requirements.txt
```

---

## Usage  
1. **Prepare Data**: Place your input files (CSV) under the `data/` directory. Each CSV should contain a **text** column for the article content and a **date** column (formatted as `YYYY-MM-DD`).  
2. **Run the Script**:  
   ```bash
   python neighborhood_analysis.py
   ```
3. **Output**:  
   - Sentiment scores for individual articles are saved in the `Sentiment_Scores_Final/` directory.  
   - Yearly sentiment scores are saved in `Date_Scores_Final/`.  
   - Yearly sentiment counts are exported to `yearly_counts.csv`.

---

## Example Input CSV Structure  
| text                                          | date       |
|-----------------------------------------------|------------|
| "Midtown is vibrant, but can be noisy."       | 2022-04-12 |
| "Englewood has improved significantly."       | 2023-08-22 |

---

## Results  
The analysis will generate:  
1. **Aggregated Sentiment Scores**: Average sentiment values across articles for each neighborhood.  
2. **Year-Based Sentiment Trends**: Allows users to compare sentiment across different years.  
3. **Graphs**: Optional bar charts to visualize trends.

---

## Contributing  
Contributions are welcome! Feel free to submit issues or pull requests to enhance the functionality of this project.  

---

## License  
This project is open-source and licensed under the MIT License.

---

## Contact  
If you have questions or feedback, feel free to reach out or open an issue on GitHub.  

---

Happy analyzing! 🎉
