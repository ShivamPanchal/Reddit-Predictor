#############################
#                           #
#    LINEAR REGRESSION      #
#                           #
#############################
#  PREDICTIVE MODELLING     #  
#############################

# Machine Learning: LINEAR REGRESSION
# Shivam Panchal


### Exploring and preparing the data
dataset <- read.csv("AmericanPolitics.csv", stringsAsFactors = FALSE, header = TRUE)
str(dataset)
# 'data.frame':	997 obs. of  22 variables::
colnames(dataset)
head(dataset)
summary(dataset$score)
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 6.00   12.00   14.00   15.87   18.00   75.00

dataset$edited <- as.factor(Archaeology$edited)
str(dataset)
table(dataset$edited)

# Because the mean value is greater than the median, 
# this implies that the distribution of insurance charges is right-skewed. 
# We can confirm this visually using a histogram:
hist(dataset$score, main = "Histogram of Score feature in the dataset", xlab = "Score of Subreddit", border = "black", col= c("red", "orange"))

table(dataset$score)


# Vizualization of Data and its features.
library(ggplot2)
ggplot(dataset,aes(x=score,y=ups)) + 
  theme(panel.background = element_rect(fill = "gray98"),
        axis.line   = element_line(colour="black"),
        axis.line.x = element_line(colour="gray"),
        axis.line.y = element_line(colour="gray")) +
  geom_point(size=2) + 
  labs(title = "Subreddit Score Vs. Ups")

library(ggplot2)
ggplot(dataset,aes(x=score,y=downs)) + 
  theme(panel.background = element_rect(fill = "gray98"),
        axis.line   = element_line(colour="black"),
        axis.line.x = element_line(colour="gray"),
        axis.line.y = element_line(colour="gray")) +
  geom_point(size=2) + 
  labs(title = "Subreddit Score Vs. Downs")

library(ggplot2)
ggplot(dataset,aes(x=score,y=num_comments)) + 
  theme(panel.background = element_rect(fill = "gray98"),
        axis.line   = element_line(colour="black"),
        axis.line.x = element_line(colour="gray"),
        axis.line.y = element_line(colour="gray")) +
  geom_point(size=2) + 
  labs(title = "Subreddit Score Vs. Number of Comments")

#  Plot determing the significant features
plot(dataset[dataset$score,c(2,7,8,9)], col= "lightgreen")



# Obtaining the correlation between the different features
cor(dataset[c("score", "ups")])

cor(dataset[c("score", "downs")])

cor(dataset[c("score", "num_comments")])

cor(dataset[c("score", "created_utc")])






# Create Training and Test data -
# set.seed(10)  # setting seed to reproduce results of random sampling
# trainingRowIndex <- sample(1:nrow(dataset), 0.75*nrow(dataset))  # row indices for training data
# training_Data <- dataset[trainingRowIndex, ]  # model training data
# test_Data  <- dataset[-trainingRowIndex, ]   # model testing data

# Let's do the cross validation with 10 folds.

#Randomly shuffle the data
Data<-dataset[sample(nrow(dataset)),]

#Create 10 equally size folds
folds <- cut(seq(1,nrow(dataset)),breaks=10,labels=FALSE)

#Perform 10 fold cross validation
for(i in 1:10){
  #Segement your data by fold using the which() function 
  testIndexes <- which(folds==i,arr.ind=TRUE)
  training_Data <- dataset[testIndexes, ]
  test_Data <- dataset[-testIndexes, ]
  #Use the test and train data partitions however you desire...
}

head(test_Data)

prediction_model <- lm(score ~ ups + downs + num_comments + edited + over_18 + title, data = training_Data )
summary(prediction_model)

par(mfrow=c(2,2))
plot(prediction_model)

prediction_model <- lm(log(score) ~ ups + num_comments, data = training_Data )
summary(prediction_model)$r.squared

par(mfrow=c(2,2))
plot(prediction_model)
