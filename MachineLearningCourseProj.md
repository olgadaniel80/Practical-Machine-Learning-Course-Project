Overview
--------

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now
possible to collect a large amount of data about personal activity
relatively inexpensively. These type of devices are part of the
quantified self movement - a group of enthusiasts who take measurements
about themselves regularly to improve their health, to find patterns in
their behavior, or because they are tech geeks. One thing that people
regularly do is quantify how much of a particular activity they do, but
they rarely quantify how well they do it. In this project, we will use
data from accelerometers on the belt, forearm, arm, and dumbell of 6
participants. They were asked to perform barbell lifts correctly and
incorrectly in 5 different ways. Our goal of the project is to predict
the manner in which they did the exercise. This is the "classe" variable
in the training set. We should create a report, answering the following
questions:

    * how we built our model, 
    * how we used cross validation, 
    * what we think the expected out of sample error is, 
    * and why we made the choices we did

Data Load And Clean
-------------------

    library(caret)

    ## Warning: package 'caret' was built under R version 3.5.2

    ## Loading required package: lattice

    ## Loading required package: ggplot2

    ## Warning: package 'ggplot2' was built under R version 3.5.1

    library(rattle)

    ## Warning: package 'rattle' was built under R version 3.5.2

    ## Rattle: A free graphical interface for data science with R.
    ## Version 5.2.0 Copyright (c) 2006-2018 Togaware Pty Ltd.
    ## Type 'rattle()' to shake, rattle, and roll your data.

    library(randomForest)

    ## Warning: package 'randomForest' was built under R version 3.5.2

    ## randomForest 4.6-14

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:rattle':
    ## 
    ##     importance

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

    library(gbm)

    ## Warning: package 'gbm' was built under R version 3.5.2

    ## Loaded gbm 2.1.5

    if(!file.exists("pml-training.csv")){
        download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", 
                      destfile = "pml-training.csv")
    }
    if(!file.exists("pml-testing.csv")){
        download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", 
                      destfile = "pml-testing.csv")
    }

    training_data <- read.csv("pml-training.csv", header = TRUE, 
                              na.strings = c("NA", "", "'#DIV/0!"))
    testing_data  <- read.csv("pml-testing.csv", header = TRUE, 
                              na.strings = c("NA", "", "'#DIV/0!"))
    dim(training_data)

    ## [1] 19622   160

    dim(testing_data)

    ## [1]  20 160

The training data include 19622 observations on 160 columns. We can see
that many columns have NA or nave no values on almost every observation.
We may remove them. The first seven columns give information about the
people who did the test, and also timestamps. Also this data we may
remove.

    # First of all we remove NA data
    training_data <- training_data[, (colSums(is.na(training_data)) == 0)]
    # Here we remove first seven columns with irrelevant informtion
    training_data <- training_data[, -c(1:7)]
    # The same manipulation on testing data
    testing_data  <- testing_data[, (colSums(is.na(testing_data)) == 0)]
    testing_data  <- testing_data[, -c(1:7)]
    dim(training_data)

    ## [1] 19622    53

    dim(testing_data)

    ## [1] 20 53

Preprocessing Data
------------------

First of all we divide our training data set into two parts - for
training and for testing.

    set.seed(1234)
    inTrain <- createDataPartition(y = training_data$classe, p = 0.75, list = FALSE)
    training <- training_data[inTrain,]
    testing <- training_data[-inTrain,]

We need to examine our data for missing data, skewed variables etc, so
that they do not affect the prediction. In order to check the data on
skewed variables, we may use histogram

    hist(as.numeric(training$classe))

![](MachineLearningCourseProj_files/figure-markdown_strict/unnamed-chunk-4-1.png)

    mean(as.numeric(training$classe))

    ## [1] 2.769398

    sd(as.numeric(training$classe))

    ## [1] 1.475522

As we can see, 2.769398 and 1.4755219 are enough closed and our
histogram also looks fine, without emissions.

    near_zero_var <- nearZeroVar(training, saveMetrics=TRUE)
    near_zero_var

    ##                      freqRatio percentUnique zeroVar   nzv
    ## roll_belt             1.154887    7.75241201   FALSE FALSE
    ## pitch_belt            1.091549   11.63201522   FALSE FALSE
    ## yaw_belt              1.095745   12.37260497   FALSE FALSE
    ## total_accel_belt      1.056863    0.19703764   FALSE FALSE
    ## gyros_belt_x          1.055286    0.89006659   FALSE FALSE
    ## gyros_belt_y          1.134554    0.44843049   FALSE FALSE
    ## gyros_belt_z          1.038835    1.12107623   FALSE FALSE
    ## accel_belt_x          1.044444    1.08710423   FALSE FALSE
    ## accel_belt_y          1.107826    0.93762740   FALSE FALSE
    ## accel_belt_z          1.026316    2.00434842   FALSE FALSE
    ## magnet_belt_x         1.032727    2.03152602   FALSE FALSE
    ## magnet_belt_y         1.086777    1.97717081   FALSE FALSE
    ## magnet_belt_z         1.040000    2.95556461   FALSE FALSE
    ## roll_arm             54.063830   16.72781628   FALSE FALSE
    ## pitch_arm            82.000000   19.48634325   FALSE FALSE
    ## yaw_arm              31.370370   18.37885582   FALSE FALSE
    ## total_accel_arm       1.061162    0.44163609   FALSE FALSE
    ## gyros_arm_x           1.015424    4.30085609   FALSE FALSE
    ## gyros_arm_y           1.437659    2.52751733   FALSE FALSE
    ## gyros_arm_z           1.145503    1.58988993   FALSE FALSE
    ## accel_arm_x           1.064516    5.21810029   FALSE FALSE
    ## accel_arm_y           1.090323    3.61462155   FALSE FALSE
    ## accel_arm_z           1.053191    5.27924990   FALSE FALSE
    ## magnet_arm_x          1.046154    8.97540427   FALSE FALSE
    ## magnet_arm_y          1.076923    5.87036282   FALSE FALSE
    ## magnet_arm_z          1.048780    8.49979617   FALSE FALSE
    ## roll_dumbbell         1.050505   86.28889795   FALSE FALSE
    ## pitch_dumbbell        2.326923   84.00597907   FALSE FALSE
    ## yaw_dumbbell          1.118280   85.60945781   FALSE FALSE
    ## total_accel_dumbbell  1.102000    0.29215926   FALSE FALSE
    ## gyros_dumbbell_x      1.034091    1.61706754   FALSE FALSE
    ## gyros_dumbbell_y      1.217489    1.82769398   FALSE FALSE
    ## gyros_dumbbell_z      1.091723    1.36567468   FALSE FALSE
    ## accel_dumbbell_x      1.040984    2.79249898   FALSE FALSE
    ## accel_dumbbell_y      1.102857    3.11183585   FALSE FALSE
    ## accel_dumbbell_z      1.141304    2.72455497   FALSE FALSE
    ## magnet_dumbbell_x     1.108527    7.30398152   FALSE FALSE
    ## magnet_dumbbell_y     1.244444    5.63255877   FALSE FALSE
    ## magnet_dumbbell_z     1.062069    4.50468814   FALSE FALSE
    ## roll_forearm         11.550781   13.18793314   FALSE FALSE
    ## pitch_forearm        73.925000   18.24296779   FALSE FALSE
    ## yaw_forearm          15.481675   12.31824976   FALSE FALSE
    ## total_accel_forearm   1.117396    0.46881370   FALSE FALSE
    ## gyros_forearm_x       1.101828    1.92961000   FALSE FALSE
    ## gyros_forearm_y       1.041237    4.91914662   FALSE FALSE
    ## gyros_forearm_z       1.174785    1.97037641   FALSE FALSE
    ## accel_forearm_x       1.184615    5.31322191   FALSE FALSE
    ## accel_forearm_y       1.136986    6.66530779   FALSE FALSE
    ## accel_forearm_z       1.017699    3.83883680   FALSE FALSE
    ## magnet_forearm_x      1.015625   10.04891969   FALSE FALSE
    ## magnet_forearm_y      1.250000   12.46772659   FALSE FALSE
    ## magnet_forearm_z      1.000000   10.99334149   FALSE FALSE
    ## classe                1.469452    0.03397201   FALSE FALSE

In the sense of missing data, we are also fine. As for strongly
correlated data, we have quite a few such variables. In our example we
will looking for variables with correlation coefficient more than 0.95.
This is too high a coefficient, however when we selected a smaller
coefficient we received too much variables. And we need to do something
with them. However, within the framework of our project, we think it
will be enough to simply demonstrate this stage. So, correlation
coefficient 0.95. It is possible to combine them, this is desirable but
not necessary.

    m <- abs(cor(training[, -53]))
    diag(m) <- 0
    which(m > 0.95, arr.ind = T)

    ##                  row col
    ## total_accel_belt   4   1
    ## accel_belt_z      10   1
    ## accel_belt_x       8   2
    ## roll_belt          1   4
    ## accel_belt_z      10   4
    ## pitch_belt         2   8
    ## roll_belt          1  10
    ## total_accel_belt   4  10
    ## gyros_dumbbell_z  33  31
    ## gyros_dumbbell_x  31  33

Model choosing
--------------

We will try different models, and choose among them the one that has the
best accuracy. In order to improve the efficiency of the models we will
use cross-validation with 5 folds.

    train_control <- trainControl(method="cv", number=5)

We start with classification tree

    mod_fit_ct <- train(classe ~ ., data = training, method = "rpart", trControl = train_control)
    fancyRpartPlot(mod_fit_ct$finalModel)

![](MachineLearningCourseProj_files/figure-markdown_strict/unnamed-chunk-8-1.png)

    # prediction
    pred_ct <- predict(mod_fit_ct, testing)
    conf_matrix_ct <- confusionMatrix(testing$classe, pred_ct)
    conf_matrix_ct$overall[1]

    ## Accuracy 
    ##  0.49531

We see that accuracy of this method is very low. The next model is
random forests.

    mod_fit_rf <- train(classe ~ ., data = training, method = "rf", trControl = train_control, verbose = FALSE)
    plot(mod_fit_rf)

![](MachineLearningCourseProj_files/figure-markdown_strict/unnamed-chunk-9-1.png)

    # prediction
    pred_rf <- predict(mod_fit_rf, testing)
    conf_matrix_rf <- confusionMatrix(testing$classe, pred_rf)
    conf_matrix_rf$overall[1]

    ##  Accuracy 
    ## 0.9932708

Now we see much more interesting accuracy. And finally we try boosting
model

    mod_fit_boost <- train(classe ~ ., data = training, method = "gbm", trControl = train_control, verbose = FALSE)
    plot(mod_fit_boost)

![](MachineLearningCourseProj_files/figure-markdown_strict/unnamed-chunk-10-1.png)

    # prediction
    pred_boost <- predict(mod_fit_boost, testing)
    conf_matrix_boost <- confusionMatrix(testing$classe, pred_boost)
    conf_matrix_boost$overall[1]

    ##  Accuracy 
    ## 0.9647227

    conf_matrix_boost

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1376   11    3    3    2
    ##          B   32  898   18    1    0
    ##          C    0   27  816   10    2
    ##          D    0    3   22  775    4
    ##          E    0   19   12    4  866
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9647          
    ##                  95% CI : (0.9592, 0.9697)
    ##     No Information Rate : 0.2871          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9554          
    ##  Mcnemar's Test P-Value : 1.297e-07       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9773   0.9374   0.9369   0.9773   0.9908
    ## Specificity            0.9946   0.9871   0.9903   0.9929   0.9913
    ## Pos Pred Value         0.9864   0.9463   0.9544   0.9639   0.9612
    ## Neg Pred Value         0.9909   0.9848   0.9864   0.9956   0.9980
    ## Prevalence             0.2871   0.1954   0.1776   0.1617   0.1782
    ## Detection Rate         0.2806   0.1831   0.1664   0.1580   0.1766
    ## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
    ## Balanced Accuracy      0.9859   0.9622   0.9636   0.9851   0.9911

Conclusion
----------

As we can see, the best result was shown by random forest. In the light
of the above about highly correlated data, we would like to look at the
most important variables for this model, in order to consider the need
to get rid of such data or the model can cope with it on its own.

    varImp(mod_fit_rf)

    ## rf variable importance
    ## 
    ##   only 20 most important variables shown (out of 52)
    ## 
    ##                      Overall
    ## roll_belt             100.00
    ## pitch_forearm          59.08
    ## yaw_belt               55.64
    ## pitch_belt             44.20
    ## roll_forearm           42.86
    ## magnet_dumbbell_z      42.60
    ## magnet_dumbbell_y      41.98
    ## accel_dumbbell_y       21.44
    ## magnet_dumbbell_x      16.97
    ## accel_forearm_x        16.87
    ## roll_dumbbell          16.74
    ## magnet_belt_z          14.92
    ## magnet_forearm_z       14.37
    ## accel_dumbbell_z       13.84
    ## accel_belt_z           13.42
    ## total_accel_dumbbell   12.52
    ## magnet_belt_y          11.63
    ## gyros_belt_z           11.34
    ## yaw_arm                11.15
    ## magnet_belt_x          10.12

    accur <- postResample(pred_rf, testing$classe)
    accur[[1]]

    ## [1] 0.9932708

The last step is to apply our model to the test dataset.

    test_set_pred <- predict(mod_fit_rf, newdata = testing_data)
    test_set_pred

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E
