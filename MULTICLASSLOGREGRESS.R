userreviews <- merge(review_data_small, user_data_small, by="user_id")

# Install and load necessary libraries
if (!require("text2vec")) install.packages("text2vec")
if (!require("tm")) install.packages("tm")
if (!require("glmnet")) install.packages("glmnet")
library(text2vec)
library(tm)
library(glmnet)

set.seed(1)
userreviews <- userreviews[sample(nrow(userreviews), 100000), ]

# Split data into training (80%) and test (20%) sets
train_indices <- sample(seq_len(nrow(userreviews)), size = 0.9 * nrow(userreviews))
train_data <- userreviews[train_indices, ]
test_data <- userreviews[-train_indices, ]

# Text preprocessing for training data
train_corpus <- Corpus(VectorSource(train_data$text))
train_corpus <- tm_map(train_corpus, content_transformer(tolower))
train_corpus <- tm_map(train_corpus, removePunctuation)
train_corpus <- tm_map(train_corpus, removeWords, stopwords("en"))
train_corpus <- tm_map(train_corpus, stemDocument)

# Prepare the text data for training
prep_text_train <- sapply(train_corpus, as.character)

# Text preprocessing for test data
test_corpus <- Corpus(VectorSource(test_data$text))
test_corpus <- tm_map(test_corpus, content_transformer(tolower))
test_corpus <- tm_map(test_corpus, removePunctuation)
test_corpus <- tm_map(test_corpus, removeWords, stopwords("en"))
test_corpus <- tm_map(test_corpus, stemDocument)

# Prepare the text data for testing
prep_text_test <- sapply(test_corpus, as.character)

# Create a document-term matrix (DTM) for training data
it_train <- itoken(prep_text_train, progressbar = FALSE)
vocabulary_train <- create_vocabulary(it_train)
vectorizer_train <- vocab_vectorizer(vocabulary_train)
dtm_train <- create_dtm(it_train, vectorizer_train)

# Create a DTM for test data
it_test <- itoken(prep_text_test, progressbar = FALSE)
vocabulary_test <- create_vocabulary(it_test)
vectorizer_test <- vocab_vectorizer(vocabulary_test)
dtm_test <- create_dtm(it_test, vectorizer_test)

# Apply TF-IDF transformation to training data
tfidf_train <- TfIdf$new()
dtm_tfidf_train <- fit_transform(dtm_train, tfidf_train)

# Apply TF-IDF transformation to test data
tfidf_test <- TfIdf$new()
dtm_tfidf_test <- fit_transform(dtm_test, tfidf_test)

# Convert DTMs to sparse matrix format for training and test data
dtm_tfidf_sparse_train <- as(dtm_tfidf_train, "CsparseMatrix")
dtm_tfidf_sparse_test <- as(dtm_tfidf_test, "CsparseMatrix")

# Ensure the response variable is a factor for training and test data
response_train <- as.factor(train_data$stars)
response_test <- as.factor(test_data$stars)

# Fit the logistic regression model using glmnet on the training sparse matrix
model <- cv.glmnet(x = dtm_tfidf_sparse_train, y = response_train, family = "multinomial")

# Prediction and Model Evaluation on the test data
predictions <- predict(model, newx = dtm_tfidf_sparse_test, s = "lambda.min", type = "class")

# Creating a confusion matrix for test data
confMat <- table(Predictions = predictions, Actual = response_test)
print(confMat)

# Calculate accuracy on test data
accuracy <- sum(diag(confMat)) / sum(confMat)
print(paste("Accuracy:", accuracy))