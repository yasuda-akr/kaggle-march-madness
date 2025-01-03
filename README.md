# March Machine Learning Mania 2024: NCAA Basketball Tournament Outcome Prediction

![March Madness](https://upload.wikimedia.org/wikipedia/en/thumb/1/1f/March_Madness_logo.svg/1200px-March_Madness_logo.svg.png)

*Predicting the outcomes of the 2024 NCAA Men's and Women's Basketball Tournaments using historical data and machine learning techniques.*

## Project Overview

March Machine Learning Mania 2024 is a Kaggle competition focused on predicting the outcomes of both the men's and women's 2024 NCAA Basketball Tournaments. The goal is to submit a portfolio of brackets based on historical game data, team statistics, seed information, and public rankings. The submissions are evaluated using the **Average Brier Bracket Score**, which assesses the accuracy of predicted probabilities for each round's outcomes.

This project leverages various data science techniques, including feature engineering, machine learning modeling, hyperparameter optimization, and simulation to generate accurate and diverse tournament brackets.

## Data Description

The dataset for this competition includes a comprehensive collection of historical NCAA basketball game results and related information, spanning both men's and women's Division I teams. Key components of the dataset include:

- **Team Information**: Unique identifiers for each team, along with their historical division statuses.
- **Game Results**: Detailed records of regular season and tournament games, including scores, locations, and overtime periods.
- **Tournament Seeds**: Seed information for each team participating in past tournaments.
- **Rankings**: Weekly team rankings from various public rating systems (e.g., Pomeroy, Sagarin, RPI).
- **Additional Data**: Conference affiliations, head coach information, alternative team name spellings, and secondary tournament results.

For a detailed schema and description of each data file, refer to the [Kaggle competition data page](#).

## Features and Methodology

### Elo Rating System

An **Elo rating** system is implemented to dynamically assess team strengths based on historical game outcomes. Each team starts with an initial rating, and ratings are updated after each game depending on the match result and the expected outcome. This system captures the relative strengths of teams over time.

### Ratings Percentage Index (RPI)

The **Ratings Percentage Index (RPI)** is calculated to measure a team's performance based on their own winning percentage, their opponents' winning percentage, and their opponents' opponents' winning percentage. This metric provides a balanced view of a team's performance relative to the competition.

### Seed Information

Tournament seed data is processed to understand the initial matchups and expected performance based on historical seed advantages. Seed differences between teams are incorporated as features to aid in predicting game outcomes.

### Massey Ordinals

**Massey Ordinals** from public ranking systems are integrated to provide additional insights into team strengths. These rankings are averaged and used as features in the predictive models.

### Modeling Approach

A **LightGBM regressor** is employed to predict the probability of one team winning against another (WinRatio). **Optuna** is used for hyperparameter optimization to enhance model performance. The model is trained on engineered features derived from Elo ratings, RPI, seeds, and rankings.

After training, the model predicts WinRatio for all possible team matchups, which are then used to simulate tournament brackets through probabilistic sampling. A large number of brackets (up to 100,000) are generated to create a diverse portfolio of predictions.

