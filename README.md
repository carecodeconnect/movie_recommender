# Movie Recommender System

## Overview
This repository contains the implementation of a movie recommendation system developed using the MovieLens 10M dataset. The system aims to provide accurate movie recommendations based on user preferences and historical ratings data.

## Dataset
The project utilizes the MovieLens 10M dataset, comprising 10 million movie ratings from various users. The dataset is split into two subsets:
- **Training Set (`edx`)**: Contains 90% of the data, used for training the model.
- **Testing Set (`final_holdout_test`)**: Comprises the remaining 10% of the data, used for evaluating the model's performance.

## Methodology
Initially planned to implement the k-nearest neighbors algorithm, the project shifted towards using Ridge regression for the recommendation system. This approach involves leveraging user and movie IDs to predict movie ratings.

### Training
The model was trained on the `edx` subset, ensuring it learns to predict ratings based on a substantial amount of data.

### Evaluation
The performance of the trained model was evaluated on the `final_holdout_test` set.

## Results
The Ridge regression model achieved a Root Mean Square Error (RMSE) of approximately 0.864 on the final holdout test set. This result indicates a relatively low average deviation of the predicted ratings from the actual ratings, demonstrating the model's effectiveness in making accurate predictions.

## Files in Repository
- `movie_recommender.pdf`: Detailed report on the development and evaluation of the recommendation system.
- `movie_recommender.R`: R script containing the code for the recommendation system.
- `movie_recommender.Rmd`: R Markdown document for the project.
- `README.md`: This document, providing an overview of the project.

## Contributions
We welcome contributions to this project! If you have suggestions or improvements, feel free to fork this repository and submit a pull request.

## License
MIT: https://mit-license.org/
