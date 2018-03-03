library(tidyverse)
library(lubridate)
library(plyr)
library(dplyr)
library(DescTools)
library(ggplot2)
df <- read_csv("loan.csv")
loan_data <- df %>%
  select(loan_status, issue_d, loan_amnt, emp_title, emp_length, verification_status, 
         home_ownership, annual_inc, purpose, inq_last_6mths, desc,
         open_acc, pub_rec, revol_util, dti, total_acc, delinq_2yrs, earliest_cr_line, mths_since_last_delinq)

#PreAnalysis Visualization
#Loan Composition
Desc(loan_data$loan_amnt, main = "Loan amount distribution", plotit = TRUE)
df$issue_d <- as.Date(gsub("^", "01-", df$issue_d), format="%d-%b-%Y")
amnt_df <- df %>%select(issue_d, loan_amnt) %>% group_by(issue_d) %>%summarise(Amount = sum(loan_amnt))

#Growth in Loans awarded by Data
ts_amnt <- ggplot(amnt_df, aes(x = issue_d, y = Amount))
ts_amnt + geom_line() + xlab("Date issued")

#loans status (Dependent Variable)
Desc(df$loan_status, plotit = T)

#Loans status by amount
box_status <- ggplot(df,aes(loan_status, loan_amnt))
box_status + geom_boxplot(aes(fill = loan_status)) +
  theme(axis.text.x = element_blank()) +
  labs(list(
    title = "Loan amount by status",
    x = "Status",
    y = "Amount"))  


loan_data2 <- df %>%
  mutate(default = ifelse(loan_status=="Charged Off", 1, 0),
         issue_d = mdy(issue_d),
         earliest_cr_line = mdy(earliest_cr_line),
         time_history = issue_d - earliest_cr_line,
         revol_util = as.numeric(sub("%","", revol_util)), 
         emp_listed = as.numeric(!is.na(emp_title) * 1),
         empty_desc = as.numeric(is.na(desc)), 
         emp_na = ifelse(emp_length == "n/a", 1, 0),
         emp_length = ifelse(emp_length == "< 1 year" | emp_length == "n/a", 0, emp_length),
         emp_length = as.numeric(gsub("\\D", "", emp_length)),
         delinq_ever = as.numeric(!is.na(mths_since_last_delinq)),   
         home_ownership = ifelse(home_ownership == "NONE", "OTHER", home_ownership) ) %>%  
  select(default, loan_amnt, empty_desc, emp_listed, emp_na, emp_length, verification_status, 
         home_ownership, annual_inc, purpose, time_history, inq_last_6mths, 
         open_acc, pub_rec, revol_util, dti, total_acc, delinq_2yrs, delinq_ever)

head(cbind(loan_data2$issue_d, loan_data2$earliest_cr_line) )
loan_data <- loan_data %>%
  mutate(issue_d = mdy(issue_d),
         earliest_cr_line = mdy(earliest_cr_line),
         time_history = issue_d - earliest_cr_line) 
head(loan_data$time_history)

loan_data <- loan_data %>%
  mutate(revol_util = as.numeric(sub("%","", revol_util)) )
head(loan_data$revol_util)


head(cbind(loan_data$emp_title, loan_data$desc, loan_data$mths_since_last_delinq))


loan_data <- loan_data %>%
  mutate(emp_listed = as.numeric(!is.na(emp_title) * 1),
         empty_desc = as.numeric(is.na(desc)), 
         delinq_ever = as.numeric(!is.na(mths_since_last_delinq)) ) 
head(cbind(loan_data$emp_listed, loan_data$empty_desc, loan_data$delinq_ever))


loan_data <- loan_data %>%
  mutate(emp_na = ifelse(emp_length == "n/a", 1, 0),
         emp_length = ifelse(emp_length == "< 1 year" | emp_length == "n/a", 0, emp_length),
         emp_length = as.numeric(gsub("\\D", "", emp_length)) )

loan_data <- loan_data %>%
  mutate(home_ownership = ifelse(home_ownership == "NONE", "OTHER", home_ownership) ) %>%  
  select(loan_status, loan_amnt, empty_desc, emp_listed, emp_na, emp_length, verification_status, 
         home_ownership, annual_inc, purpose, time_history, inq_last_6mths, 
         open_acc, pub_rec, revol_util, dti, total_acc, delinq_2yrs, delinq_ever)
str(loan_data)
loan_data2$time_history=loan_data$time_history
clean_data <- as.data.frame(model.matrix(~ ., data = loan_data2))[, -1]
str(clean_data)
clean_data<-clean_data[,-c(14:26)]
# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(clean_data$default, SplitRatio = 0.75)
training_set = subset(clean_data, split == TRUE)
test_set = subset(clean_data, split == FALSE)


# Feature Scaling
training_set[-1] = scale(training_set[-1])
test_set[-1] = scale(test_set[-1])

training_set$default=as.factor(training_set$default)

# Fitting Logistic Regression to the Training set
classifier = glm(formula = default ~ .,
                 family = binomial,
                 data = training_set)

# Predicting the Test set results
prob_pred = predict(classifier, type = 'response', newdata = test_set[-1])
y_pred = ifelse(prob_pred > 0.5, 1, 0)

# Making the Confusion Matrix
cm = table(test_set[, 1], y_pred > 0.5)


# load the library
library(mlbench)
library(caret)

# define the control using a random forest selection function
control <- rfeControl(functions=rfFuncs, method="cv", number=2)
# run the RFE algorithm
results <- rfe(training_set[,-1], training_set[,1],size=25,rfeControl=control)
# summarize the results
print(results)
# list the chosen features
predictors(results)