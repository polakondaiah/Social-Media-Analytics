"""
Social Media Analytics Project
Name:
Roll Number:
"""
from fileinput import filename
from tkinter import CENTER
import hw6_social_tests as test

project = "Social" # don't edit this

### PART 1 ###

import pandas as pd
import nltk
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
from collections import Counter
import operator
endChars = [ " ", "\n", "#", ".", ",", "?", "!", ":", ";", ")" ]

'''
makeDataFrame(filename)
    #3 [Check6-1]
Parameters: str
Returns: dataframe
'''
def makeDataFrame(filename):
    df = pd.read_csv(filename)
    
    return df


'''
parseName(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''
def parseName(fromString):
   # print(fromString)
    start = fromString.find(":") + \
        len(":")
    end = fromString.find("(")
    final = fromString[start:end].strip()
    return final


'''
parsePosition(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''
def parsePosition(fromString):
    start = fromString.find("(") + \
        len(":")
    end = fromString.find("from")
    final = fromString[start:end].strip()
    
    return final


'''
parseState(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''
def parseState(fromString):
    cut = fromString.find("(")
    sub_s = fromString[cut:]
    start = sub_s.find("from") + \
    len("from")
    end = sub_s.find(")")
    final = sub_s[start:end].strip()
    #print(sub_s)
    return final


'''
findHashtags(message)
#5 [Check6-1]
Parameters: str
Returns: list of strs
'''
def findHashtags(message):
    words_list = message.split("#")
   # print(words_list)
    hashtag_words = []
    for word in words_list[1:]:
        hash_word = ""
        for letter in word:
            if letter not in endChars:
                hash_word += letter
            else:
                break
        hashtag_words.append("#"+hash_word)
               
       
    return hashtag_words


'''
getRegionFromState(stateDf, state)
#6 [Check6-1]
Parameters: dataframe ; str
Returns: str
'''
def getRegionFromState(stateDf, state):
    reg = stateDf.loc[stateDf['state'] == state, 'region']
    return reg.values[0]


'''
addColumns(data, stateDf)
#7 [Check6-1]
Parameters: dataframe ; dataframe
Returns: None
'''
def addColumns(data, stateDf):
    names_l = []
    positions_l = []
    states_l = []
    regions_l = []
    hashtags_l = []
   
    for index, row in data.iterrows():
        a = row["label"]
        v = row["text"]
        name = parseName(a)
        names_l.append(name)
        position = parsePosition(a)
        positions_l.append(position)      
        state = parseState(a)
        states_l.append(state)
        region = getRegionFromState(stateDf,state)
        regions_l.append(region)
        hashtags = findHashtags(v)
        hashtags_l.append(hashtags)
    data["name"] = names_l
    data["position"] = positions_l
    data["state"] = states_l
    data["region"] = regions_l
    data["hashtags"] = hashtags_l
              
       
    return None


### PART 2 ###

'''
findSentiment(classifier, message)
#1 [Check6-2]
Parameters: SentimentIntensityAnalyzer ; str
Returns: str
'''
def findSentiment(classifier, message):
    
    score = classifier.polarity_scores(message)['compound']
    if score<(-0.1):
        return "negative"
    elif score>0.1:
        return "positive"
    else:
        return "neutral"


'''
addSentimentColumn(data)
#2 [Check6-2]
Parameters: dataframe
Returns: None
'''
def addSentimentColumn(data):
    classifier = SentimentIntensityAnalyzer()
    sentiments_l = []
    for index,row in data.iterrows():
        mes = row["text"]
        sentiment = findSentiment(classifier,mes)
        sentiments_l.append(sentiment)
    data["sentiment"] = sentiments_l
    return None


'''
getDataCountByState(data, colName, dataToCount)
#3 [Check6-2]
Parameters: dataframe ; str ; str
Returns: dict mapping strs to ints
'''
def getDataCountByState(data, colName, dataToCount):
    dict_map = {}
    for index, row in data.iterrows():
        if len(colName) != 0 and len(dataToCount) != 0:
            if row[colName] == dataToCount:
                if row["state"] in dict_map:
                    dict_map[row["state"]] += 1
                else:
                    dict_map[row["state"]] = 1
              
        else:
            if row["state"] in dict_map:
                dict_map[row["state"]] += 1
            else:
                dict_map[row["state"]] = 1
   # print(dict_map)
    return dict_map


'''
getDataForRegion(data, colName)
#4 [Check6-2]
Parameters: dataframe ; str
Returns: dict mapping strs to (dicts mapping strs to ints)
'''
def getDataForRegion(data, colName):
    outer_dict = {}
    for index, row in data.iterrows():
        if row["region"] not in outer_dict:
            outer_dict [row["region"]] = {}
             
            
        if row [colName] in outer_dict[row["region"]]:
            outer_dict[row["region"]][row[colName]] +=1
        else:
            outer_dict[row["region"]][row[colName]] =1
           
    return outer_dict


'''
getHashtagRates(data)
#5 [Check6-2]
Parameters: dataframe
Returns: dict mapping strs to ints
'''
def getHashtagRates(data):
    hash_dict = {}
    for index,row in data.iterrows():
        for hashtag in row["hashtags"]:
            if hashtag in hash_dict:
                hash_dict[ hashtag] += 1
            else:
                hash_dict[ hashtag] = 1
            
    return hash_dict


'''
mostCommonHashtags(hashtags, count)
#6 [Check6-2]
Parameters: dict mapping strs to ints ; int
Returns: dict mapping strs to ints
'''
def mostCommonHashtags(hashtags, count):
    sort_dict = dict((sorted(hashtags.items(), key= lambda x : x[1], reverse = True))[:count])
    #dict_map = [(i,j) for i,j in sort_dict.items()]
   # print(sort_dict)
    return sort_dict

'''
getHashtagSentiment(data, hashtag)
#7 [Check6-2]
Parameters: dataframe ; str
Returns: float
'''
def getHashtagSentiment(data, hashtag):
    t_mes = 0
    count = 0
    for index, row in data.iterrows():
        if hashtag in findHashtags(row["text"]):
            if row["sentiment"] == "positive":
                count += 1
            elif row["sentiment"] == "negative":
                count -= 1
            elif row["sentiment"] == "neutral":
                count += 0
            t_mes += 1
            
    
    return count/t_mes


### PART 3 ###

'''
graphStateCounts(stateCounts, title)
#2 [Hw6]
Parameters: dict mapping strs to ints ; str
Returns: None
'''
def graphStateCounts(stateCounts, title):
    import matplotlib.pyplot as plt
    dict_to_list = list(stateCounts.items())
    for key,value in dict_to_list:
        labels = key
        yValues = value
        plt.bar(labels,yValues,color='red')
        plt.xlabel(title,loc='center')
        plt.xticks(rotation="vertical")
        plt.title(title)
        
    plt.show()
    
    return


'''
graphTopNStates(stateCounts, stateFeatureCounts, n, title)
#3 [Hw6]
Parameters: dict mapping strs to ints ; dict mapping strs to ints ; int ; str
Returns: None
'''
def graphTopNStates(stateCounts, stateFeatureCounts, n, title):
    
    
    feature ={}
    h = {}
    #print(stateCounts)
    #print(stateFeatureCounts)
    for i in stateFeatureCounts:
        feature[i] = stateFeatureCounts[i]/stateCounts[i]
    #print(feature)
    s = Counter(feature)
    #print(s)
    sort = list(sorted(s.items(), key=operator.itemgetter(1),reverse=True))[:n]
    for key,value in sort:
        h[key] = value
    graphStateCounts(h,title)
    return


'''
graphRegionComparison(regionDicts, title)
#4 [Hw6]
Parameters: dict mapping strs to (dicts mapping strs to ints) ; str
Returns: None
'''
def graphRegionComparison(regionDicts, title):
   # print(regionDicts)
    region_names = []
    feature_names = []
    feature_values = []
    for key,values in regionDicts.items():
        #print(key)
        # print(values)
        region_names.append(key)
        temp = []   
        for feature,count in values.items():
            if feature not in feature_names:
                feature_names.append(feature)
            temp.append(count)
            #print(temp)
        #print(temp)
        #print(feature_names)
        feature_values.append(temp)
    #print(region_feature)
    sideBySideBarPlots(feature_names,region_names,feature_values,title)
    
    return


'''
graphHashtagSentimentByFrequency(data)
#4 [Hw6]
Parameters: dataframe
Returns: None
'''
def graphHashtagSentimentByFrequency(data):
    # hashtags = []
    # frequency = []
    # sentiment = []
    # hashtag = getHashtagRates(data)
    # common = mostCommonHashtags(hashtag,50)
    # for key,value in common.items():
    #     hashtags.append(key)
    #    # if value not in frequency
    #     frequency.append(value)
    #     sentiment.append(getHashtagSentiment(data,key))
    # scatterPlot(frequency,sentiment,hashtags,"sentiment_graph")
    return


#### PART 3 PROVIDED CODE ####
"""
Expects 3 lists - one of x labels, one of data labels, and one of data values - and a title.
You can use it to graph any number of datasets side-by-side to compare and contrast.
"""
def sideBySideBarPlots(xLabels, labelList, valueLists, title):
    import matplotlib.pyplot as plt

    w = 0.8 / len(labelList)  # the width of the bars
    xPositions = []
    for dataset in range(len(labelList)):
        xValues = []
        for i in range(len(xLabels)):
            xValues.append(i - 0.4 + w * (dataset + 0.5))
           
        xPositions.append(xValues)
        # print(xValues)

    for index in range(len(valueLists)):
        plt.bar(xPositions[index], valueLists[index], width=w, label=labelList[index])

    plt.xticks(ticks=list(range(len(xLabels))), labels=xLabels, rotation="vertical")
    plt.legend()
    plt.title(title)

    plt.show()

"""
Expects two lists of probabilities and a list of labels (words) all the same length
and plots the probabilities of x and y, labels each point, and puts a title on top.
Expects that the y axis will be from -1 to 1. If you want a different y axis, change plt.ylim
"""
def scatterPlot(xValues, yValues, labels, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    plt.scatter(xValues, yValues, color = 'hotpink')

    # make labels for the points
    for i in range(len(labels)):
        plt.annotate(labels[i], # this is the text
                    (xValues[i], yValues[i]), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0, 10), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center

    plt.title(title)
    plt.ylim(-1, 1)

    # a bit of advanced code to draw a line on y=0
    ax.plot([0, 1], [0.5, 0.5], color='black', transform=ax.transAxes)

    plt.show()
 

### RUN CODE ###

# This code runs the test cases to check your work
if __name__ == "__main__":
  #  print("\n" + "#"*15 + " WEEK 1 TESTS " +  "#" * 16 + "\n")
   # test.week1Tests()
  #  print("\n" + "#"*15 + " WEEK 1 OUTPUT " + "#" * 15 + "\n")
    # test.runWeek1()
    # test.testMakeDataFrame()
    # test.testParseName()
    # test.testParsePosition()
    # test.testParseState()
    # test.testFindHashtags()
    # test.testGetRegionFromState()
    
    ## Uncomment these for Week 2 ##
 #   print("\n" + "#"*15 + " WEEK 2 TESTS " +  "#" * 16 + "\n")
  #  test.week2Tests()
 #   print("\n" + "#"*15 + " WEEK 2 OUTPUT " + "#" * 15 + "\n")
   # test.runWeek2()
    # df = makeDataFrame("data/politicaldata.csv")
    # test.testFindSentiment()
    # test.testAddSentimentColumn()
    # test.testGetDataCountByState(df)
    # test.testGetHashtagRates()
    # test.testGetHashtagSentiment()

    ## Uncomment these for Week 3 ##
    
    print("\n" + "#"*15 + " WEEK 3 OUTPUT " + "#" * 15 + "\n")
    test.runWeek3()
    
