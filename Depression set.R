#Make sure you have java installed on your computer
#install.packages('h2o')
library(h2o)
library(lime)
h2o.init() #Starts up H2O cluster/server locally 

#This is where we prepare and collect the data to insert in into the model
data <- read.csv(file.choose())
data <- na.omit(data)
data <- as.data.frame(data)
data <- as.h2o(data) #Sending data to the H20 server
cols <- c(6,11,12,15,18)
yes_or_no<- setdiff(1:31,cols)

for(col in yes_or_no){
  data[,col] <- as.factor(data[,col])
}
splits <- h2o.splitFrame(data,0.8) #Splitting data in train and test with 80/20
train <- splits[[1]]
test <- splits[[2]]
cols_x <- setdiff(colnames(train),c("Depression",
                                    "I.identify.as.having.a.mental.illness",
                                    "Anxiety",
                                    "Obsessive.thinking",
                                    "Mood.swings",
                                    "Panic.attacks",
                                    "Lack.of.concentration",
                                    "Tiredness",
                                    "I.have.been.hospitalized.before.for.my.mental.illness",
                                    "How.many.days.were.you.hospitalized.for.your.mental.illness",
                                    "Device.Type",
                                    "Compulsive.behavior",
                                    "How.many.times.were.you.hospitalized.for.your.mental.illness"
                                    ))
cols_y <- "Depression"
h2o.getTypes(train)
colnames(train)
#Used AutoML for inspiration
GBM <- h2o.gbm(cols_x,cols_y,train,model_id = "Depression_data",
                  nfolds = 5, 
                  fold_assignment = "Stratified", 
                  stopping_metric	= "logloss",
                  ntrees = 37,
                  max_depth = 7,
                  min_rows = 15,
                  stopping_rounds = 5,
                  stopping_tolerance = 0.05,
                  distribution = "bernoulli",
                  max_runtime_secs = 300,
                  col_sample_rate_per_tree = 0.4,
                  sample_rate = 0.9,
                  col_sample_rate = 0.7,
                  seed = 123
                  )
#We used stopping_rounds, stopping_tolerance, stopping_metrics, col_sample_rate_per_tree
#to prevent overfitting
GBM@model_id

#Results
GBM
h2o.performance(GBM,test)
result <- h2o.predict(GBM, test)
resultDF <- as.data.frame(result)
test1 <- as.data.frame(test)
no_medical <- test1[,c('Education',"Region","Household.Income","Age","Gender",)]
resultDF <- cbind(resultDF['predict'],no_medical)
plot(h2o.performance(GBM),type = "roc")
for(i in seq_along(1:length(resultDF))){
  if(resultDF$predict[i] != test1$Depression[i])
  {
    resultDF$predict[i] <- NA
  }
}
resultDF <- na.omit(resultDF)

#The commented part below is demostrating the use of LIME for analysis
#explainer <- lime(test1, GBM, bin_continuous = TRUE, quantile_bins = FALSE)
# explanation <- explain(test1, explainer, n_labels = 2, n_features = 7) #nlabels = 2 because there are two possible outcome, True or False.
# explanation[,2:9][1:5,]
# plot_features(explanation[explanation$case==1,], ncol = 1)
