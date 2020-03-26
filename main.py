import utils as ns
import time
import matplotlib.pyplot as plt
import numpy as np
import string
# import pandas as pd
# import random

englishTrain, englishTrainAlphabet = ns.loadTextData("data/english.test")

englishTest, englishTestAlphabet = ns.loadTextData("data/english.train")


def test_r_chunk():
    nRange = np.arange(1, 1000, 100)
    # sampleSizeList = [1.0, 0.75, 0.5, 0.25, 0.1]
    sampleRList = [4, 5, 6, 7, 8]
    plt.figure()
    for s in sampleRList:
        x = []
        timeList = []
        for n in nRange:
            x.append(n)
            startTime = time.time()
            detectors = ns.trainRChunk(englishTrain,
                                       englishTrainAlphabet,
                                       n,
                                       int(1.0 * len(englishTrain)),
                                       s,
                                       len(englishTrain[0]))
            endTime = time.time()
            timeList.append(endTime - startTime)
        plt.plot(x, timeList, label=str(s))
    plt.title("r-chunk training time")
    plt.xlabel("Number of Detectors Trained")
    plt.ylabel("time(seconds)")
    # plt.legend(title="Fraction of Training Strings Tested")
    plt.legend(title="Varying r values")
    plt.savefig("plots/r_chunkTrainTime_line_n_r.jpg")


def test_r_contiguous():
    """
    Tests for R_contiguous
    :return:
    """
    nRange = np.arange(1, 1000, 100)
    # sampleSizeList = [1.0, 0.75, 0.5, 0.25, 0.1]
    sampleRList = [4, 5, 6, 7, 8]
    plt.figure()
    for s in sampleRList:
        x = []
        timeList = []
        for n in nRange:
            x.append(n)

            startTime = time.time()

            detectors = ns.trainRContig(englishTrain,
                                        englishTrainAlphabet,
                                        n,
                                        int(1.0 * len(englishTrain)),
                                        s,
                                        len(englishTrain[0]))

            endTime = time.time()
            timeList.append(endTime - startTime)
        plt.plot(x, timeList, label=str(s))
    plt.title("r-contiguous training time")
    plt.xlabel("Number of Detectors Trained")
    plt.ylabel("time(seconds)")
    # plt.legend(title="Fraction of Training Strings Tested")
    plt.legend(title="Varying r values")
    plt.savefig("plots/r_contiguousTrainTime_line_n_r.jpg")


def classifying_languages():
    r = 4
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
    anomaly_scores = [[], [], [], []]

    n = 100000

    result = "Name\t" + \
             "repSize\t" + \
             "time\t" + \
             "count\t" + \
             "truePerc\t" + \
             "AnomalyScore\t" + \
             "Threshold\t" + \
             "FP\t" + \
             "TP_Above\t" + \
             "TP_Below\t"
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

            anomaly_score = 1
            flag = False

            for d in detectors:
                if d.testDetector(testString):
                    if not flag:
                        anomalies += 1
                    flag = True
                    anomaly_score += 1
            anomaly_scores[i].append(np.log10(anomaly_score))

        n_normal_above_threshold = 0
        n_anomaly_above_threshold = 0
        n_anomaly_below_threshold = 0
        threshold = 0.5

        for anomaly_score in anomaly_scores[i]:
            if anomaly_score == 0.0:
                n_normal_above_threshold += 1
            elif 0 < anomaly_score < threshold:
                n_anomaly_below_threshold += 1
            else:
                n_anomaly_above_threshold += 1

        result = "" + languageNames[i] + "\t" + \
                 str(n) + "\t" + str(deltaTime) + "\t" + \
                 str(anomalies) + "/" + str(numTrueAnomalies[i]) + "\t" + \
                 str(float(anomalies) / float(numTrueAnomalies[i])) + "\t" + \
                 str(sum(anomaly_scores[i])) + "\t" + \
                 str(threshold) + "\t" + \
                 str(float(n_normal_above_threshold)/float(numTrueAnomalies[i])) + "\t" + \
                 str(float(n_anomaly_above_threshold)/float(numTrueAnomalies[i])) + "\t" + \
                 str(float(n_anomaly_below_threshold)/float(numTrueAnomalies[i]))
        result = result.expandtabs(20)
        print(result)


def classify_cardio():
    r = 6
    a = 10
    n_detectors = 10000
    alphabets = list(string.ascii_lowercase[:a])
    numTrueAnomalies = 0
    anomalies = 0
    anomaly_scores = []

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
    # print(deltaTime)

    n_normal_above_threshold = 0
    n_normal = 0
    n_anomaly = 0
    n_anomaly_above_threshold = 0
    n_anomaly_below_threshold = 0
    threshold = 0.5

    for testString in cardioTestData:
        if testString[:-1] not in cardioTrainData:
            numTrueAnomalies += 1

        anomaly_score = 1
        flag = False

        if testString[-1] == "f":
            n_normal += 1
        else:
            n_anomaly += 1

        for d in detectors:
            if d.testDetector(testString):
                if not flag:
                    anomalies += 1
                flag = True
                anomaly_score += 1
        anomaly_scores.append(np.log10(anomaly_score))
    print(f"Normal count: {n_normal} Anomaly Count: {n_anomaly}")

    for i, anomaly_score in enumerate(anomaly_scores):
        if cardioTestData[i][-1] == 'f' and anomaly_score > threshold:
            n_normal_above_threshold += 1
        elif cardioTestData[i][-1] != 'f' and anomaly_score < threshold:
            n_anomaly_below_threshold += 1
        elif cardioTestData[i][-1] != 'f' and anomaly_score >= threshold:
            n_anomaly_above_threshold += 1

    result = "Name\t" + \
             "repSize\t" + \
             "time\t" + \
             "count\t" + \
             "truePerc\t" + \
             "AnomalyScore\t" + \
             "Threshold\t" + \
             "FP\t" + \
             "TP_Above\t" + \
             "TP_Below\t"
    result = result.expandtabs(20)
    print(result)

    result = "Cardio - Time" + "\t" + \
             str(n_detectors) + "\t" + str(deltaTime) + "\t" + \
             str(anomalies) + "/" + str(numTrueAnomalies) + "\t" + \
             str(float(anomalies) / float(numTrueAnomalies)) + "\t" + \
             str(sum(anomaly_scores)) + "\t" + \
             str(threshold) + "\t" + \
             str(float(n_normal_above_threshold) / float(n_normal)) + "\t" + \
             str(float(n_anomaly_above_threshold) / float(n_anomaly)) + "\t" + \
             str(float(n_anomaly_below_threshold) / float(n_anomaly))
    result = result.expandtabs(20)
    print(result)


def classify_newdataset():
    r = 5
    n_detectors = 10000
    numTrueAnomalies = 0
    anomalies = 0
    anomaly_scores = []

    datasc_train, alphabets = ns.loadTextData("data/datasc23.train")
    # alphabets, _ = ns.loadTextData("data/datasc23.alpha")
    # alphabets = list(alphabets[0])
    datasc_test, _ = ns.loadTextData("data/datasc23.3.test")
    datasc_label, _ = ns.loadTextData("data/datasc23.3.labels")

    startTime = time.time()
    detectors = ns.trainRChunk(datasc_train,
                                alphabets,
                                n_detectors,
                                int(len(datasc_train)),
                                r,
                                len(datasc_train[0]))
    endTime = time.time()
    deltaTime = endTime - startTime
    # print(deltaTime)

    n_normal_above_threshold = 0
    n_normal = 0
    n_anomaly = 0
    n_anomaly_above_threshold = 0
    n_anomaly_below_threshold = 0
    threshold = 0.3

    for i, testString in enumerate(datasc_test):
        if testString not in datasc_label:
            numTrueAnomalies += 1
        anomaly_score = 1
        flag = False

        if datasc_label[i] == '0':
            n_normal += 1
        else:
            n_anomaly += 1

        for d in detectors:
            if d.testDetector(testString):
                if not flag:
                    anomalies += 1
                flag = True
                anomaly_score += 1
        anomaly_scores.append(np.log10(anomaly_score))

    print(f"Normal count: {n_normal} Anomaly Count: {n_anomaly}")

    for i, anomaly_score in enumerate(anomaly_scores):
        if anomaly_score > threshold and datasc_label[i] == '0':
            n_normal_above_threshold += 1
        elif anomaly_score < threshold and datasc_label[i] == '1':
            n_anomaly_below_threshold += 1
        elif datasc_label[i] == '1':
            n_anomaly_above_threshold += 1

    result = "Name\t" + \
             "repSize\t" + \
             "time\t" + \
             "count\t" + \
             "truePerc\t" + \
             "AnomalyScore\t" + \
             "Threshold\t" + \
             "FP\t" + \
             "TP_Above\t" + \
             "TP_Below\t"
    result = result.expandtabs(20)
    print(result)

    result = "Datasc23" + "\t" + \
             str(n_detectors) + "\t" + str(deltaTime) + "\t" + \
             str(anomalies) + "/" + str(numTrueAnomalies) + "\t" + \
             str(float(anomalies) / float(numTrueAnomalies)) + "\t" + \
             str(sum(anomaly_scores)) + "\t" + \
             str(threshold) + "\t" + \
             str(float(n_normal_above_threshold) / float(n_normal)) + "\t" + \
             str(float(n_anomaly_above_threshold) / float(n_anomaly)) + "\t" + \
             str(float(n_anomaly_below_threshold) / float(n_anomaly))
    result = result.expandtabs(20)
    print(result)


# print(max_val, min_val)

# test_r_chunk()
# test_r_contiguous()
classifying_languages()
# ns.check_for_split(data)
# classify_cardio()
# classify_newdataset()
