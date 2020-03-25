import utils as ns
import time
import matplotlib.pyplot as plt
import numpy as np
import string
# import pandas as pd
# import random

# load english train and test data
englishTrain, englishTrainAlphabet = ns.loadTextData("data/english.test")

englishTest, englishTestAlphabet = ns.loadTextData("data/english.train")


# do some time testing for r-chunk
def test_r_chunk():
    nRange = np.arange(1, 1000, 100)
    sampleSizeList = [1.0, 0.75, 0.5, 0.25, 0.1]
    plt.figure()
    for s in sampleSizeList:
        x = []
        timeList = []
        for n in nRange:
            x.append(n)
            startTime = time.time()
            detectors = ns.trainRChunk(englishTrain,
                                       englishTrainAlphabet,
                                       n,
                                       int(s*len(englishTrain)),
                                       7,
                                       len(englishTrain[0]))
            endTime = time.time()
            timeList.append(endTime - startTime)
        plt.scatter(x, timeList, label=str(s))
    plt.title("Naive r-chunk training time")
    plt.xlabel("Number of Detectors Trained")
    plt.ylabel("Wall time(seconds)")
    plt.legend(title="Fraction of Training Strings Tested")
    plt.savefig("plots/r_chunkTrainTime.jpg")


def test_r_contiguous():
    """
    Tests for R_contiguous
    :return:
    """
    nRange = np.arange(1, 1000, 100)
    sampleSizeList = [1.0, 0.75, 0.5, 0.25, 0.1]
    plt.figure()
    for s in sampleSizeList:
        x = []
        timeList = []
        for n in nRange:
            x.append(n)

            startTime = time.time()

            detectors = ns.trainRContig(englishTrain,
                                        englishTrainAlphabet,
                                        n,
                                        int(s * len(englishTrain)),
                                        4,
                                        len(englishTrain[0]))

            endTime = time.time()
            timeList.append(endTime - startTime)
        plt.scatter(x, timeList, label=str(s))
    plt.title("Naive r-contiguous training time")
    plt.xlabel("Number of Detectors Trained")
    plt.ylabel("Wall time(seconds)")
    plt.legend(title="test")
    plt.savefig("plots/r_contiguousTrainTime.jpg")


def classifying_languages():
    r = 7
    hilTest, hilAlphabet = ns.loadTextData("data/hiligaynon.txt")
    midTest, midAlphabet = ns.loadTextData("data/middle-english.txt")
    dietschTest, dietshAlphabet = ns.loadTextData("data/plautdietsch.txt")
    xhosaTest, xhosaAlphabet = ns.loadTextData("data/xhosa.txt")
    globalAlphabet = []
    globalAlphabet.extend(hilAlphabet)
    globalAlphabet.extend(midAlphabet)
    globalAlphabet.extend(dietshAlphabet)
    globalAlphabet.extend(xhosaAlphabet)
    globalAlphabetSet = set(globalAlphabet)
    globalAlphabet = list(globalAlphabetSet)

    languageNames = ["Hiligaynon", "Middle-English", "Plautdietsch", 'Xhosa']
    languageData = [hilTest, midTest, dietschTest, xhosaTest]
    numTrueAnomalies = [0, 0, 0, 0]
    englishSet = set(englishTrain)

    n = 10000
    r = 4

    result = "Name\t" + \
             "repSize\t" + \
             "time\t" + \
             "count\t" + \
             "truePerc"
    result = result.expandtabs(20)
    print(result)

    startTime = time.time()
    detectors = ns.trainRContig(englishTrain,
                                globalAlphabet,
                                n,
                                int(len(englishTrain)),
                                r,
                                10)
    endTime = time.time()
    deltaTime = endTime - startTime

    for i in range(len(languageData)):

        # train a reprtoire
        # test each string in current test set
        anomalies = 0
        for testString in languageData[i]:
            if testString not in englishSet:
                numTrueAnomalies[i] += 1

            for d in detectors:
                if d.testDetector(testString):
                    anomalies += 1
                    break
        result = "" + languageNames[i] + "\t" + \
                 str(n) + "\t" + str(deltaTime) + "\t" + \
                 str(anomalies) + "/" + str(numTrueAnomalies[i]) + "\t" + str(
            float(anomalies) / float(numTrueAnomalies[i]))
        result = result.expandtabs(20)
        print(result)


def classify_cardio():
    r = 5
    a = 10
    n_detectors = 10000
    alphabets = list(string.ascii_lowercase[:a])
    numTrueAnomalies = 0
    anomalies = 0

    max_val, min_val, cardioTrain = ns.loadCSVData("data/cardio_train.csv")
    max_val, min_val, cardioTest = ns.loadCSVData("data/cardio_test.csv")

    cardioTrainData = ns.binned_data(cardioTrain, a, max_val, min_val)
    cardioTestData = ns.binned_data(cardioTest, a, max_val, min_val)
    print("Binning Complete")
    startTime = time.time()
    detectors = ns.trainRContig(cardioTrain,
                                alphabets,
                                n_detectors,
                                int(len(cardioTrain)),
                                r,
                                len(cardioTrainData[0]))
    endTime = time.time()
    deltaTime = endTime - startTime
    print(deltaTime)
    for testString in cardioTestData:
        if testString not in cardioTrainData:
            numTrueAnomalies += 1

        for d in detectors:
            if d.testDetector(testString):
                anomalies += 1
                break
    print(f"Cardio - Time: {deltaTime} Anamolies={anomalies}/{numTrueAnomalies} TruePrec: {anomalies/numTrueAnomalies}")
    pass


# print(max_val, min_val)

# test_r_chunk()
# test_r_contiguous()
# classifying_languages()
# ns.check_for_split(data)

classify_cardio()
