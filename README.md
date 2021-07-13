### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>
Python version used: 3.8.7

Packages used:
1. Pandas 1.2.3
2. Matplotlib 3.4.1
3. Numpy 1.20.2
4. Plotly 4.14.3
5. Scikit-Learn 0.24.1
6. Scipy 1.6.2

Datasets used in the analysis are included in this repo, for convenience ("data" directory). Data are simulated, and files are quite small.

## Project Motivation<a name="motivation"></a>
I completed this project as the capstone of Udacity's Data Scientist Nanodegree Program in July of 2021.

The primary motivation of this project was to practice concepts learned in the program:
1. Python class development
2. Machine learning (unsupervised clustering with K Means)
3. Exploratory data analysis
4. Strategic data cleaning (i.e. with an eye toward a strategic business question)
5. Git repo management
6. Statistical significance testing

Secondarily, my motivation was to answer the business question I came up with after exploring Starbucks's simulated data:

> "To which user segment(s) should each of Starbucks's promotion types be targeted?"

## File Descriptions <a name="files"></a>

- Starbucks_Capstone_notebook.ipynb: The principal file in this repo. It is the report that documents my process, my results and what I believe are their implications.
- binomial.py: A custom class I wrote that allowed me to evaluate my results for significance very quickly.
- portfolio.json: A small dataset describing each offer in Starbucks's promotional repertoire.
- profile.json: Dataset of Starbucks's user base, described by gender, age, income and the date they became a member.
- transcript.json: Dataset of all transactions taking place among users during the simulated experiment's time frame. Events recorded include offer receipts, offer views, offer completions and monetary transactions.

## Results<a name="results"></a>
Results are described in detail in the "Project Summary" section of the Jupyter notebook.

In preview, this analysis aggregated Starbucks users into 5 clusters:
1. Low Earners
2. High Earners
3. Male Middle Earners
4. Female Middle Earners
5. The Unknowns

It then found that most groups prefer Starbucks's discount promo offer, while the unknowns don't respond to either.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Data were simulated and made available by [Starbucks](https://www.starbucks.com). Feel free to use the code provided in this repository at your own discretion.
