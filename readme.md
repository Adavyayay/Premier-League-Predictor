# ⚽ Premier League Match Outcome Predictor

This project predicts the outcome of **English Premier League football matches** (Home Win / Draw / Away Win) using **historical match results** and a **Random Forest classifier**.  

It downloads raw match data from [football-data.co.uk](https://www.football-data.co.uk/), cleans and processes it, engineers rolling team performance features, and trains a machine learning model.


## 📌 Key Idea
Team winning does not just depend upon the matches won or lost, but on **team strength** and **form in recent matches**.

## 📊 Model Performance
While the model does not have very high accuracy overall, testing on **Matchdays 1 and 2** gave promising results:  

- Out of **10 matches**, the model predicted **8 correctly** ✅


## 🔮 Upcoming Match Predictions

### Liverpool vs Arsenal (31 Aug 2025)
| Outcome        | Probability |
|----------------|-------------|
| **Arsenal Win** | 0.200 |
| **Draw**        | 0.110 |
| **Liverpool Win** | 0.690 |


### Man City vs Man United (14 Sept 2025)
| Outcome        | Probability |
|----------------|-------------|
| **Man United Win** | 0.270 |
| **Draw**           | 0.150 |
| **Man City Win**   | 0.580 |



---
## 📬 Connect with Me
- **LinkedIn:** [Adavya Goel](https://www.linkedin.com/in/adavyagoel/)  
- **GitHub:** [Adavyayay](https://github.com/Adavyayay)  
- **Email:** adavya601@gmail.com  


⭐ If you like this project, don’t forget to **star** the repo!

---

## 📊 Data

We use **English Premier League match data** from `football-data.co.uk`.  
For each season, the dataset contains many columns, including odds and statistics.  
We keep only the **core match outcome fields**:

| Column       | Meaning                                |
|--------------|----------------------------------------|
| `season`     | Season (e.g. 2023-24)                  |
| `date`       | Match date                             |
| `home_team`  | Home team name                         |
| `away_team`  | Away team name                         |
| `home_goals` | Full-time goals scored by home team    |
| `away_goals` | Full-time goals scored by away team    |
| `result`     | Full-time result: `H`=Home win, `D`=Draw, `A`=Away win |


# 🛠 Feature Engineering

To predict future matches, we compute rolling statistics from each team’s last **N = 5** matches.

## Definitions (safe, plain-text formulas)

- **Points per game (PPG)**  
  PPG = (total points in last N matches) / N  
  where points per match are: win = 3, draw = 1, loss = 0.

- **Team strength (rescaled PPG)**  
  `strength = 50 + 15 * (PPG - 1.5)`  
  *Reason:* rescales PPG (0..3) into approximately `[27.5, 72.5]`.

- **Recent form (rescaled to ~[0.5, 2.0])**  
  `form = 0.5 + (PPG / 3) * 1.5`  
  *Reason:* maps PPG=0 → 0.5, PPG=3 → 2.0.

- **Goals averages**  
  `goals_avg = total_goals_scored / N`  
  `goals_conceded_avg = total_goals_conceded / N`

- **Home advantage**  
  A constant bias currently set to `1.0` (placeholder).

## Final feature vector for each match

These 9 features are used to train the model.  
The **target variable** is the match outcome (`result`).


## 🤖 Model

We use a **Random Forest Classifier** from `scikit-learn`.

- **Train/test split**: 80/20  
- **Evaluation metrics**: Accuracy & classification report  
- **Feature importance**: Printed to understand key predictors  

---

## ▶️ How to Run

    pip install numpy pandas scikit-learn requests
    python predictor.py


Built with ❤️ using Python, pandas, and scikit-learn.

