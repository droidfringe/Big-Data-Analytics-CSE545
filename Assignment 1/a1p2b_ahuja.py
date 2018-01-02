import os
import codecs
import re
from pyspark import SparkContext, SparkConf

# Read file into string
# Take care of unicode characters
def ReadXMLFileToString(filePath):
    '''
    # Fails if there are unreadable bytes
    with open(filePath,'r') as myFile:
        data = myFile.read()
    return data
    '''
    with codecs.open(filePath, 'r', encoding='utf-8', errors='replace') as myFile:
        data = myFile.read()
    return data

# Custom Parser
def ConvertTextToWords(text):
    startTag = '<Blog>'
    endTag = '</Blog>'
    postStartTag = '<post>'
    postEndTag = '</post>'
    dateStartTag = '<date>'
    dateEndTag = '</date>'
    startIdx = 6
    endIdx = 13
    iterations = 0
    datePostTuples = []
    isEndFound = False
    while(text[endIdx-7:endIdx] != endTag):
        
        while(text[startIdx:startIdx+6] != dateStartTag):
            startIdx += 1
            if(startIdx + 6 >= len(text)):
                isEndFound = True
                break
        if(isEndFound):
            break
        endIdx = startIdx + 13
        while(text[endIdx-6:endIdx+1] != dateEndTag):
            endIdx += 1
        currPostDate = text[startIdx+6:endIdx-6]
        dateSplit = currPostDate.split(',')
        currPostDate = dateSplit[2] + '-' + dateSplit[1]
        #print(currPostDate)
        startIdx = endIdx + 1
        while(text[startIdx:startIdx+6] != postStartTag):
            startIdx += 1
        endIdx = startIdx + 13
        while(text[endIdx-6:endIdx+1] != postEndTag):
            endIdx += 1
        currPost = text[startIdx+6:endIdx-6]
        #print(currPost)
        datePostTuples.append((currPostDate, currPost))
        startIdx = endIdx + 1
        #iterations += 1
        #if(iterations > 5):
        #    break
    return datePostTuples

# In a blog, select only the Industry words
# Send them in a list, and associate year-month of the blog with words
def ConvertSentenceToWordsInIndustry(timestamp, sentence, industryDict):
    industryWords = []
    #for word in sentence.split(' '):
    for word in re.split(' |,|\.|:|;|\\n|\\r|"|-|\'|\?', sentence):
        #print(word)
        if(word.lower() in industryDict):
            industryWords.append(((word.lower(), timestamp),1))
    return industryWords

# Main spark rdd transformations
def CountOccuranceOfIndustriesInAllFiles(sc, datasetDir, industryDict, numFiles=None):
    filesList = os.listdir(datasetDir)
    if(numFiles == None):
        numFiles = len(filesList)
    data = sc.parallelize(filesList[0:numFiles])
    industryDictBC = sc.broadcast(list(industryDict))
    dataTransformation = data.map(lambda k: datasetDir + k).map(ReadXMLFileToString).flatMap(ConvertTextToWords)
    out = dataTransformation.flatMap(lambda k: ConvertSentenceToWordsInIndustry(k[0], k[1], industryDictBC.value)).reduceByKey(lambda a,b: a+b).map(lambda k: (k[0][0], ((k[0][1],k[1]),))).reduceByKey(lambda a,b: a+(b)).collect()
    return out

if __name__ == "__main__":
    sc = SparkContext.getOrCreate()
    # Don't forget to include / at the end of datasetDir
    datasetDir = '/home/gaurav/Acads/CSE545/blogs/'

    # RDD Transformations to get list of industries
    filesList = os.listdir(datasetDir)
    data = sc.parallelize(filesList)

    industryList = data.map(lambda k: (k.split('.')[-3],1)).reduceByKey(lambda a,b: a).map(lambda k: k[0]).collect()
    industryDict = dict()
    for industry in industryList:
        industryDict[industry.lower()] = True

    industryDictBC = sc.broadcast(list(industryDict))
    datasetDirBC = sc.broadcast(str(datasetDir))
    industryCounts = CountOccuranceOfIndustriesInAllFiles(sc, datasetDir, industryDict)
    print(industryCounts)

'''
Industries with 0 count:

lawenforcement-security
investmentbanking
museums-libraries
sports-recreation
communications-media
indunk
businessservices
humanresources
'''

