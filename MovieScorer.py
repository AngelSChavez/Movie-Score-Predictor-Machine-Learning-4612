import csv
import re
import nltk.corpus
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

#This is the TF-IDF algorithm that will calculate the TF-IDF values of the movies
tfidfAlgorithm = TfidfVectorizer(use_idf = True)

#List where tweets will be housed
tweetTrainingData = []
tweetTestingData = []

#List where scores will be housed
scoreData = []
testScoreData = []

#Words to exclude from datasets
wordsToRemove = stopwords.words('english')
excludeTitle = ["title"]
wordsToRemove.extend(excludeTitle)

#This is the csv reader that will read the training data
with open("tweets_training.csv") as Tweets:
    tweetReader = csv.reader(Tweets)

    for row in tweetReader:

        #Text extracted from dataset
        tweetsBeingRead = row[0]

        #Tweets set to lower case, @'s and hyperlinks removed, and all symbols removed
        tweetsBeingRead = tweetsBeingRead.lower()
        tweetsBeingRead = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", tweetsBeingRead)

        #Stopwords removed from tweets and movie title removed from tweet
        tweetsBeingRead = " ".join([word for word in tweetsBeingRead.split() if word not in (wordsToRemove)])

        #Tweets split into individual tweets
        tweetsBeingRead = tweetsBeingRead.split("\n")

        #Every tweet that is processed will become am element in this list
        tweetTrainingData.extend(tweetsBeingRead)

        #The scores are stored on the second column of the csv file
        Scores = row[1]

        #The scores are added to the total data scores
        scoreData.append(Scores)

#This will read the tweets that will have their scores predicted
with open("tweets_test.csv") as Tweets:
    tweetReader = csv.reader(Tweets)

    for row in tweetReader:

        #Text extracted from dataset
        tweetsBeingRead = row[0]

        #Tweets set to lower case, @'s and hyperlinks removed, and all symbols removed
        tweetsBeingRead = tweetsBeingRead.lower()
        tweetsBeingRead = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", tweetsBeingRead)

        #Stopwords removed from tweets and movie title removed from tweet
        tweetsBeingRead = " ".join([word for word in tweetsBeingRead.split() if word not in (wordsToRemove)])

        #Tweets split into individual tweets
        tweetsBeingRead = tweetsBeingRead.split("\n")

        #Every tweet that is processed will become am element in this list
        tweetTestingData.extend(tweetsBeingRead)

        #These are the actual scores of the movies for comparison's sake
        testScores = row[1]
        testScoreData.append(testScores)


#The algorithm is processing and organizing the training data to create the TF-IDF values of the tweets
tfidfValues = tfidfAlgorithm.fit_transform(tweetTrainingData)
tfidfValues = tfidfAlgorithm.transform(tweetTrainingData)
tfidfTestValues = tfidfAlgorithm.fit_transform(tweetTestingData)
tfidfTestValues = tfidfAlgorithm.transform(tweetTestingData)
tfidfTestValues.shape

#These are the words that have been assigned value
wordsOfValue = tfidfAlgorithm.get_feature_names_out()

#Naive Bayes Classifier is used to assign the TF-IDF to the scores of the movies
naiveBayesClassifier = MultinomialNB()
naiveBayesClassifier.fit(tfidfValues, scoreData)
naiveBayesClassifier.fit(tfidfTestValues, testScoreData)

#The predictions are made for the specific values
finalPrediction = naiveBayesClassifier.predict(tfidfTestValues)

#Thesse values and lists will hose the number of total correct and total predicted scores
realScoreCategories = []
veryGoodRealScore = 0
goodRealScore = 0
mediocreRealScore = 0
badRealScore = 0

predictedScoreCategories = []
veryGoodPredictedScore = 0
goodPredictedScore = 0
mediocrePredictedScore = 0
badPredictedScore = 0

#This list of lists is required to organize the predicted IMDB scores in a manner which properly exports to CSV
printableValues = [[]]

#With this for loop and these if statement chains, the program will be able to create lists of all the values separated
finalPredictionLength = len(finalPrediction)

for i in range(finalPredictionLength):
    value1 = float(finalPrediction[i])
    value2 = float(testScoreData[i])
    printableValues.append([str(value1)])

    if (value1 < 6.0):
        badRealScore += 1
    elif ((value1 >= 6.0) & (value1 < 7.0)):
        mediocreRealScore += 1
    elif ((value1 >= 7.0) & (value1 < 8.0)):
        goodRealScore += 1
    else:
        veryGoodRealScore += 1

    if (value2 < 6.0):
        badPredictedScore += 1
    elif ((value2 >= 6.0) & (value2 < 7.0)):
        mediocrePredictedScore += 1
    elif ((value2 >= 7.0) & (value2 < 8.0)):
        goodPredictedScore += 1
    else:
        veryGoodPredictedScore += 1


printableValues.pop(0)
realCategories = [veryGoodRealScore, goodRealScore, mediocreRealScore, badRealScore]
predictedCategories = [veryGoodPredictedScore, goodPredictedScore, mediocrePredictedScore, badPredictedScore]

print("Actual Test Score Totals (Very Good, Good, Mediocre, Bad):\n")
print(realCategories)
print("\n")
print("Predicted Test Score Totals (Very Good, Good, Mediocre, Bad):\n")
print(predictedCategories)
print("\n")

#These are the classification report (precision, recall, f1-score, support) and the confusion matrix
print("Classification Report:\n")
#There will be a few warnings when this prints, this is due to the scores occassionally being missing
print(metrics.classification_report(testScoreData, finalPrediction))
print("Confusion Matrix:\n")
print(metrics.confusion_matrix(testScoreData, finalPrediction))

#This output file holds all the predicted values
outputFile = "output.csv"

with open(outputFile, "w", newline='') as results:
    outputWriter = csv.writer(results)
    outputWriter.writerows(printableValues)