
##########################################################
# Create edx and final_holdout_test sets 
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(knitr)
library(ggplot2)
library(RColorBrewer)
library(dplyr)
library(hexbin)
library(glmnet)
library(Matrix)
library(tidyverse)
library(caret)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Number of rows
nrow_edx <- nrow(edx)

# Number of columns
ncol_edx <- ncol(edx)

nrow_edx
ncol_edx

# Number of zeros given as ratings
n_zeros <- sum(edx$rating == 0)

# Number of threes given as ratings
n_threes <- sum(edx$rating == 3)

n_zeros
n_threes

# Number of different movies
n_movies <- length(unique(edx$movieId))

n_movies

# Number of different users
n_users <- length(unique(edx$userId))


# Summary Statistics Table
summary_stats <- data.frame(
  Statistic = c("Number of Rows", "Number of Columns", "Number of Zeros (Ratings)", 
                "Number of Threes (Ratings)", "Number of Different Movies", 
                "Number of Different Users"),
  Value = c(nrow_edx, ncol_edx, n_zeros, n_threes, n_movies, n_users)
)
kable(summary_stats, caption = "Summary Statistics of the MovieLens Dataset")

# Number of movie ratings in each genre
n_drama <- sum(grepl("Drama", edx$genres))
n_comedy <- sum(grepl("Comedy", edx$genres))
n_thriller <- sum(grepl("Thriller", edx$genres))
n_romance <- sum(grepl("Romance", edx$genres))

list(
  Number_of_Users = n_users,
  Ratings_in_Drama = n_drama,
  Ratings_in_Comedy = n_comedy,
  Ratings_in_Thriller = n_thriller,
  Ratings_in_Romance = n_romance
)

# Genre Ratings Bar Plot
genre_ratings <- data.frame(
  Genre = c("Drama", "Comedy", "Thriller", "Romance"),
  Ratings = c(n_drama, n_comedy, n_thriller, n_romance)
)
ggplot(genre_ratings, aes(x = Genre, y = Ratings)) + 
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(title = "Number of Movie Ratings per Genre", 
       x = "Genre", y = "Number of Ratings")

# Count the number of ratings for each movie
movie_rating_counts <- edx %>%
  group_by(title) %>%
  summarise(number_of_ratings = n()) %>%
  ungroup()

# Find the movie with the greatest number of ratings
top_rated_movie <- movie_rating_counts %>%
  arrange(desc(number_of_ratings)) %>%
  top_n(1)

top_rated_movie

# Top Rated Movie
kable(top_rated_movie, caption = "Most Rated Movie in the Dataset")

# Count the number of times each rating was given
rating_counts <- table(edx$rating)

# Sort the ratings in descending order to get the most given ratings
sorted_rating_counts <- sort(rating_counts, decreasing = TRUE)

# Get the names of the top five ratings
top_five_ratings <- names(sorted_rating_counts)[1:5]

top_five_ratings

# Ratings Distribution
rating_distribution <- as.data.frame(table(edx$rating))
ggplot(rating_distribution, aes(x = Var1, y = Freq)) +
  geom_bar(stat = "identity", fill = "coral") +
  labs(title = "Distribution of Movie Ratings", 
       x = "Rating", y = "Frequency")

# Count the number of times each rating was given
rating_counts <- table(edx$rating)

# Convert to data frame for easier manipulation
rating_counts_df <- as.data.frame(rating_counts)

# Separate whole and half star ratings
whole_star_ratings <- rating_counts_df[grep("\\.0", rating_counts_df$Var1), ]
half_star_ratings <- rating_counts_df[grep("\\.5", rating_counts_df$Var1), ]

# Check if, in general, half star ratings are less common than whole star ratings
less_common_half_stars <- all(half_star_ratings$Freq < whole_star_ratings$Freq)

less_common_half_stars

# Checking if half star ratings are less common
half_star_comparison <- data.frame(
  Rating_Type = c("Whole Star Ratings", "Half Star Ratings"),
  Less_Common = c(any(whole_star_ratings$Freq > half_star_ratings$Freq), less_common_half_stars)
)


# Setting seed for reproducibility
set.seed(123)

# Splitting the edx dataset into training and test sets
splitIndex <- createDataPartition(edx$rating, p = 0.8, list = FALSE, times = 1)
trainSet <- edx[splitIndex,]
testSet <- edx[-splitIndex,]

# Combine train and test sets to ensure consistent factor levels
combinedSet <- rbind(trainSet, testSet)

# Creating sparse matrices for user and movie factors with consistent levels
user_matrix <- sparse.model.matrix(~ as.factor(userId) - 1, data = combinedSet)
movie_matrix <- sparse.model.matrix(~ as.factor(movieId) - 1, data = combinedSet)

# Splitting the combined matrix back into train and test
n_train <- nrow(trainSet)
combined_matrix <- cbind(user_matrix, movie_matrix)
train_matrix <- combined_matrix[1:n_train, ]
test_matrix <- combined_matrix[(n_train + 1):nrow(combined_matrix), ]

# Training the Ridge regression model with glmnet
set.seed(123) # for reproducibility
ridge_model <- glmnet(train_matrix, trainSet$rating, alpha = 0)

# Making predictions on the test set
predictions <- predict(ridge_model, s = 0.01, newx = test_matrix)

# Calculating RMSE
rmse_value <- sqrt(mean((testSet$rating - predictions)^2))

# Output the RMSE
print(paste("RMSE on the test set:", rmse_value))

kable(half_star_comparison, caption = "Comparison of Whole vs Half Star Ratings")

# Preparing the final_holdout_test set in the same format as the training data
# This is crucial to ensure consistency in factor levels between the datasets

# Combining final_holdout_test with the combinedSet to align factor levels
combinedSetForHoldout <- rbind(combinedSet, final_holdout_test)

# Creating sparse matrices for user and movie factors with consistent levels
holdout_user_matrix <- sparse.model.matrix(~ as.factor(userId) - 1, data = combinedSetForHoldout)
holdout_movie_matrix <- sparse.model.matrix(~ as.factor(movieId) - 1, data = combinedSetForHoldout)

# Extracting the holdout matrix
n_combined <- nrow(combinedSet)
holdout_matrix <- cbind(holdout_user_matrix, holdout_movie_matrix)[(n_combined + 1):nrow(combinedSetForHoldout), ]

# Making predictions on the final_holdout_test set
final_predictions <- predict(ridge_model, s = 0.01, newx = holdout_matrix)

# Calculating RMSE for the final_holdout_test set
final_rmse <- sqrt(mean((final_holdout_test$rating - final_predictions)^2))

# Output the RMSE for the final_holdout_test set
print(paste("RMSE on the final_holdout_test set:", final_rmse))

# First, check the structure of final_holdout_test and final_predictions
print(head(final_holdout_test))
print(length(final_predictions))

# Convert final_predictions to a vector if it's not already
final_predictions_vector <- as.vector(final_predictions)

# Add the predicted ratings to the final_holdout_test data frame
final_holdout_test$predicted_rating <- final_predictions_vector

# Assuming 'rating' in final_holdout_test is the actual rating
# and we have already added a 'predicted' column from final_predictions_vector
full_data <- final_holdout_test %>%
  mutate(predicted = final_predictions_vector)

# Testing different sample sizes to see how they affect representativeness
sample_sizes <- c(1000, 5000, 10000)
samples <- lapply(sample_sizes, function(size) {
  full_data %>%
    group_by(rating) %>%
    sample_n(size = min(size, n()), replace = TRUE) %>%
    ungroup()
})

# Choose the sample size that provides a good balance between speed and representativeness
representative_sample <- samples[[2]] # assuming the second sample size was chosen

ggplot(representative_sample, aes(x = rating, y = predicted)) +
  geom_jitter(alpha = 0.2, width = 0.2, height = 0) +
  geom_smooth(method = 'lm', formula = y ~ x, color = "red") +
  labs(title = "Actual vs. Predicted Ratings with Linear Model Smooth",
       x = "Actual Rating", 
       y = "Predicted Rating") +
  theme_minimal()

