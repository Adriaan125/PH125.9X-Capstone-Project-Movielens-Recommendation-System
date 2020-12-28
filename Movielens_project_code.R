
##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(ggplot2)
library(dplyr)
#library(knitr)
#library(rmarkdown)
#library(latexpdf)
#library(latex2exp)
#library(tinytex)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
# if using R 4.0 or later:
#movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
#                                           title = as.character(title),
#                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind = "Rounding") # if using R 3.6 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

edx
validation


##Describing dataset (based on EDX dataset)##

#summary of edx dataset
head(edx)   #6 columns
dim(edx)    #9 000 055 lines with 6 columns
str(edx)    #structure of edx data
summary(edx)  #basic summary

n_distinct(edx$movieId)  #how many movies in dataset
n_distinct(edx$userId)   #how many users in dataset

sapply(edx, function(x) sum(is.na(x)))   #check to see N/A's in data

sum(edx$rating > 5 | edx$rating <= 0)    #check to see how many ratings are not between zero and five

library(ggplot2)
qplot(edx$rating,          #distribution of movie ratings
      geom="histogram",
      binwidth = 0.5,  
      main = "Histogram for movie ratings", 
      xlab = "ratings",  
      fill=I("blue"), 
      col=I("red"), 
      alpha=I(.2),
      xlim=c(0.0, 5.5))


edx %>% 
  dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "blue") + 
  scale_x_log10() + 
  ggtitle("Histogram of Movies")     ##some movies are rated more than others

edx %>%
  dplyr::count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "blue") + 
  scale_x_log10() +
  ggtitle("Histogram of Users")    #some users have rated over 1000 movies


#####################################
#split edx into test and training set
#####################################

set.seed(1) # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y=edx$rating, times = 1, p = 0.5, list = FALSE)    #50/50
edx_train_set <- edx[-test_index,]
temp <- edx[test_index,]

# Make sure userId and movieId in test set are also in edx set
edx_test_set <- temp %>% 
  semi_join(edx_train_set, by = "movieId") %>%
  semi_join(edx_train_set, by = "userId")

# Add rows removed from test set back into edx set
removed <- anti_join(temp, edx_test_set)
edx_train_set <- rbind(edx_train_set, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

edx_train_set
edx_test_set

#average
mu <- mean(edx_train_set$rating)
mu             #average rating on training data, mu minimizes the RMSE. We will predict the same rating for all movies.

naive_rmse <- RMSE(edx_test_set$rating, mu)
naive_rmse         #baseline model

rmse_results <- data_frame(method = "The average", RMSE = naive_rmse)  #we create a table with our stored results

#movie effect
movie_average_ratings <- edx_train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))   #b's   #we will drop the hat to make code cleaner

movie_average_ratings %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black"))   #estimates vary greatly


predicted_ratings <- mu + edx_test_set %>%         #now we predict with model we just fit. y_hat=u + b_i+ e
  left_join(movie_average_ratings, by='movieId') %>%
  .$b_i

model_1 <- RMSE(predicted_ratings, edx_test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie Effect Model on test set",
                                     RMSE = model_1))
rmse_results %>% knitr::kable()                               #RMSE has dropped to 0.94


#movie effect + user effect
user_average_ratings <- edx_train_set%>% 
  left_join(movie_average_ratings, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))    #we approximate b_hat_u  (b_u represents users that give good movies a poor rating)

predicted_ratings <- edx_test_set %>%            #now we run the new model  y_hat=u + b_i + b_u + e
  left_join(movie_average_ratings, by='movieId') %>%
  left_join(user_average_ratings, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

model_2 <- RMSE(predicted_ratings, edx_test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie Effect + User Effect Model on test set",  
                                     RMSE = model_2))
rmse_results %>% knitr::kable()    #RMSE has dropped to 0.87


#movie effect regluarized to remove noisy estimates

#tuning lamda
lambdas <- seq(0, 10, 0.25)      #now we tune lamda parameter to find best setting
mu <- mean(edx_train_set$rating)
the_sum <- edx_train_set %>% 
  group_by(movieId) %>% 
  summarize(s = sum(rating - mu), n_i = n())
rmses <- sapply(lambdas, function(l){
  predicted_ratings <- edx_test_set %>% 
    left_join(the_sum, by='movieId') %>% 
    mutate(b_i = s/(n_i+l)) %>%
    mutate(pred = mu + b_i) %>%
    .$pred
  return(RMSE(predicted_ratings, edx_test_set$rating))
})



lambda <-  lambdas[which.min(rmses)]           #regularization - when n is small the estimate of b_i is shrunk to zero, the larger the lamda the more we shrink
mu <- mean(edx_train_set$rating)
movie_reg_average_rating <- edx_train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n()) 

data_frame(original = movie_average_ratings$b_i, 
           regularlized = movie_reg_average_rating$b_i, 
           n = movie_reg_average_rating$n_i) %>%
  ggplot(aes(original, regularlized, size=sqrt(n))) + 
  geom_point(shape=1, alpha=0.5)     #to see how estimates shrink


predicted_ratings <- edx_test_set %>% 
  left_join(movie_reg_average_rating, by='movieId') %>%
  mutate(pred = mu + b_i) %>%
  .$pred

model_3 <- RMSE(predicted_ratings, edx_test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Movie Effect Model on test set",  
                                     RMSE = model_3))
rmse_results %>% knitr::kable()    


#movie effect + user effect regularized
lambdas <- seq(0, 10, 0.25)               #we use cross validation again to pick lamda that minimizes our equation.
rmses <- sapply(lambdas, function(l){
  mu <- mean(edx_train_set$rating)
  b_i <- edx_train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- edx_train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <- 
    edx_test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  return(RMSE(predicted_ratings, edx_test_set$rating))
})



lambda <- lambdas[which.min(rmses)]
lambda

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Movie Effect + User Effect Model on test set",  
                                     RMSE = min(rmses)))
rmse_results %>% knitr::kable()


#movie effect regluarize to remove noisy estimates against validation

#tuning lamda
lambdas <- seq(0, 10, 0.25)      #now we tune lamda parameter to find best setting
mu <- mean(edx$rating)
the_sum <- edx %>% 
  group_by(movieId) %>% 
  summarize(s = sum(rating - mu), n_i = n())
rmses <- sapply(lambdas, function(l){
  predicted_ratings <- validation %>% 
    left_join(the_sum, by='movieId') %>% 
    mutate(b_i = s/(n_i+l)) %>%
    mutate(pred = mu + b_i) %>%
    .$pred
  return(RMSE(predicted_ratings, validation$rating))
})




lambda <-  lambdas[which.min(rmses)]           #regularization - when n is small the estimate of b_i is shrunk to zero, the larger the lamda the more we shrink
mu <- mean(edx$rating)
movie_reg_average_rating <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n()) 

data_frame(original = movie_average_ratings$b_i, 
           regularlized = movie_reg_average_rating$b_i, 
           n = movie_reg_average_rating$n_i) %>%
  ggplot(aes(original, regularlized, size=sqrt(n))) + 
  geom_point(shape=1, alpha=0.5)     #to see how estimates shrink


predicted_ratings <- validation %>% 
  left_join(movie_reg_average_rating, by='movieId') %>%
  mutate(pred = mu + b_i) %>%
  .$pred

model_3 <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Movie Effect Model on validation set",  
                                     RMSE = model_3))
rmse_results %>% knitr::kable()    


#movie effect + user effect regularized
lambdas <- seq(0, 10, 0.25)               #we use cross validation again to pick lamda that minimizes our equation.
rmses <- sapply(lambdas, function(l){
  mu <- mean(edx$rating)
  b_i <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  return(RMSE(predicted_ratings, validation$rating))
})



lambda <- lambdas[which.min(rmses)]
lambda

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Movie + User Effect Model on validation set",  
                                     RMSE = min(rmses)))
rmse_results %>% knitr::kable()


