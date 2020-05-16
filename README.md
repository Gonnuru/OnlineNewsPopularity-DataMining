# Data Mining - Online News Popularity 

### Project Goal
Analyze the number of shares depending on the attributes and predict if an article will be popular on the internet or not.

### About Mashable
Mashable is a global, multi-platform media and entertainment company. Powered by its own proprietary technology, Mashable is the go-to source for tech, digital culture and entertainment content for its dedicated and influential audience around the globe. The data set contains details about the articles analyzed from Jan 2013 – Jan 2015

### Business Understanding
The aim of the project is to analyze the Online News Popularity data set and create a predictive model based on supervised learning. The model is used to help the news organizations identify an article that may become popular which can be a useful strategy and financial interest to websites. For instance, examining the effect of components like number of pictures, recordings, news released per day, news area, news subjectivity, popularity, can help the news distributing organization to change their news covering technique to fulfill their customer and to attract more customers.


### About dataset

https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity
The dataset contains 61 attributes out of which 58 attributes are predictive, 2 non-predictive attributes and 1 target attribute.The target attribute being the number of shares of that particular article there are various attributes taken into consideration like-number of images, number of videos,unique tokens and so on that are contributing towards the increase in the number of shares.
In the below step the dataset is loaded into the book using pandas dataframe. And all the attributes in the dataset are listed.

***Attribute Information:***
0. url: URL of the article (non-predictive)
1. timedelta: Days between the article publication and the dataset acquisition (non-predictive)
2. n_tokens_title: Number of words in the title
3. n_tokens_content: Number of words in the content
4. n_unique_tokens: Rate of unique words in the content
5. n_non_stop_words: Rate of non-stop words in the content
6. n_non_stop_unique_tokens: Rate of unique non-stop words in the content
7. num_hrefs: Number of links
8. num_self_hrefs: Number of links to other articles published by Mashable
9. num_imgs: Number of images
10. num_videos: Number of videos
11. average_token_length: Average length of the words in the content
12. num_keywords: Number of keywords in the metadata
13. data_channel_is_lifestyle: Is data channel 'Lifestyle'?
14. data_channel_is_entertainment: Is data channel 'Entertainment'?
15. data_channel_is_bus: Is data channel 'Business'?
16. data_channel_is_socmed: Is data channel 'Social Media'?
17. data_channel_is_tech: Is data channel 'Tech'?
18. data_channel_is_world: Is data channel 'World'?
19. kw_min_min: Worst keyword (min. shares)
20. kw_max_min: Worst keyword (max. shares)
21. kw_avg_min: Worst keyword (avg. shares)
22. kw_min_max: Best keyword (min. shares)
23. kw_max_max: Best keyword (max. shares)
24. kw_avg_max: Best keyword (avg. shares)
25. kw_min_avg: Avg. keyword (min. shares)
26. kw_max_avg: Avg. keyword (max. shares)
27. kw_avg_avg: Avg. keyword (avg. shares)
28. self_reference_min_shares: Min. shares of referenced articles in Mashable
29. self_reference_max_shares: Max. shares of referenced articles in Mashable
30. self_reference_avg_sharess: Avg. shares of referenced articles in Mashable
31. weekday_is_monday: Was the article published on a Monday?
32. weekday_is_tuesday: Was the article published on a Tuesday?
33. weekday_is_wednesday: Was the article published on a Wednesday?
34. weekday_is_thursday: Was the article published on a Thursday?
35. weekday_is_friday: Was the article published on a Friday?
36. weekday_is_saturday: Was the article published on a Saturday?
37. weekday_is_sunday: Was the article published on a Sunday?
38. is_weekend: Was the article published on the weekend?
39. LDA_00: Closeness to LDA topic 0
40. LDA_01: Closeness to LDA topic 1
41. LDA_02: Closeness to LDA topic 2
42. LDA_03: Closeness to LDA topic 3
43. LDA_04: Closeness to LDA topic 4
44. global_subjectivity: Text subjectivity
45. global_sentiment_polarity: Text sentiment polarity
46. global_rate_positive_words: Rate of positive words in the content
47. global_rate_negative_words: Rate of negative words in the content
48. rate_positive_words: Rate of positive words among non-neutral tokens
49. rate_negative_words: Rate of negative words among non-neutral tokens
50. avg_positive_polarity: Avg. polarity of positive words
51. min_positive_polarity: Min. polarity of positive words
52. max_positive_polarity: Max. polarity of positive words
53. avg_negative_polarity: Avg. polarity of negative words
54. min_negative_polarity: Min. polarity of negative words
55. max_negative_polarity: Max. polarity of negative words
56. title_subjectivity: Title subjectivity
57. title_sentiment_polarity: Title polarity
58. abs_title_subjectivity: Absolute subjectivity level
59. abs_title_sentiment_polarity: Absolute polarity level
60. shares: Number of shares (target)

### Data Preparation:
Data preparation is one of the important steps in data mining. The data used for this project is firstly checked for any null values and the data is checked for any noisy variables. The target variable is defined as a continuous variable int the dataset, the target is transformed into categorical to fit the classification models used for analysis.  The attributes are selected based on domain knowledge and the recursive feature elimination (RFE) model provided by the sklearn library.
The problem of data leakage is handled by using the practices that are used to minimize data leakage when developing predictive models. The data is split into two training and validation data and the data is prepared using cross validation folds
As the shares column is in the numeric type and we have to classify it into two class problem popular or unpopular.
So, we have to choose a threshold and we choose 1400 shares as the threshold and converted into binary class problem.
As **Max: 843,300, Mean: 3,395.380, Deviation: 11,626.951 Median shares: 1,400 shares,** there is a lot of deviation and choose median as it does not affect outliers.
Since, the data is highly skewed, we are using Log Transformation to make data less skewed. This can be valuable both for making patterns in the data more interpretable and for helping to meet the assumptions of inferential statistics.
Before Log transformation:
![LogTransformationPic](https://github.com/Gonnuru/OnlineNewsPopularity-DataMining/blob/master/Images/beforelog.jpg)

After Log Transformation and converting the shares variable into two categories
![AfterLogTransformation](https://github.com/Gonnuru/OnlineNewsPopularity-DataMining/blob/master/Images/afterlog.jpg)

We used **Recursive Feature Elimination (RFE)**
Recursive Feature Elimination or RFE uses a model (e.g. linear Regression or SVM) to select either the best or worst-performing feature, and then excludes that feature. The whole process is then iterated until all features in the dataset are used up (or up to a user-defined limit). Sklearn conveniently possesses a RFE function via the sklearn feature selection call and we use this along with a simple linear regression model and logistic regression model for ranking the features and to decide on the attribute for the model building.

### Model Building:
We Implemented Three Classification models. Decision Trees, Support Vector Machines and K Nearest Neighbors (KNN)
1)	**Decision Trees:**
A decision tree is a guide of the conceivable results of a progression of related choices. Decision tree is one of the most used techniques in data mining because of its simplicity to explain the results. Besides, there are decision tree algorithms that work with parallel and incremental techniques, which help to process large databases for classifying new objects faster than traditional algorithms. A decision tree ordinarily begins with a single node, which branches into conceivable results.

![Decision Tree](https://github.com/Gonnuru/OnlineNewsPopularity-DataMining/blob/master/Images/decision.jpg)

***ROC Curve for Decision Trees***

2)	**Support Vector Machines (SVM):**
A Support Vector Machine (SVM) is a discriminative classifier formally defined by a separating hyperplane. In other words, given labelled training data (supervised learning), the algorithm outputs an optimal hyperplane which categorizes new examples. In two-dimensional space this hyperplane is a line dividing a plane in two parts where in each class lay in either side.

![SVM](https://github.com/Gonnuru/OnlineNewsPopularity-DataMining/blob/master/Images/SVM.jpg)

***ROC Curve For SVM***

3)	**K Nearest Neighbors (KNN):**
K-Nearest Neighbors is one of the most basic yet essential classification algorithms in Machine Learning. It belongs to the supervised learning domain and finds intense application in pattern recognition, data mining and intrusion detection. In pattern recognition, the k-nearest neighbors’ algorithm (k-NN) is a non-parametric method used for classification and regression. In both cases, the input consists of the k closest training examples in the feature space. The output depends on whether k-NN is used for classification or regression: In k-NN classification, the output is a class membership. An object is classified by a majority vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor.

![Knearest](https://github.com/Gonnuru/OnlineNewsPopularity-DataMining/blob/master/Images/KNN.jpg)

***ROC Curve For KNN***

### Results 
![Results](https://github.com/Gonnuru/OnlineNewsPopularity-DataMining/blob/master/Images/results.jpg)
### Tableau Insights
![1](https://github.com/Gonnuru/OnlineNewsPopularity-DataMining/blob/master/Images/tableau1.jpg)
![2](https://github.com/Gonnuru/OnlineNewsPopularity-DataMining/blob/master/Images/tableau2.jpg)
![3](https://github.com/Gonnuru/OnlineNewsPopularity-DataMining/blob/master/Images/tableau3.jpg)
![4](https://github.com/Gonnuru/OnlineNewsPopularity-DataMining/blob/master/Images/tableau4.jpg)
### Conclusion
With the aim to predict the popularity of a news on the various factors we build models based on three different supervised classification techniques
- Decision Tree
-	Support Vector Machine
-	K-Nearest Neighbor

As we can observe through the results, The Support Vector Machine has performed better as compared to Decision Tree and K-Nearest Neighbor in accuracy, sensitivity and Precision
From the available **61 attributes we identified 21 based on domain knowledge and the recursive feature elimination (RFE) model**. Some of the key factors that contribute towards news popularity are
- Was the article published on the weekend?
- 	Is data channel 'Entertainment'?
- 	Is data channel 'Tech'?
-	The Pictures, Videos, rate of Positive words and rate of negative words in article.

### Output
The SVM model that we have built gives **68% accuracy** on the testing data. Thus, we have built a model that can predict if the News will become popular based on the given features of the news.









