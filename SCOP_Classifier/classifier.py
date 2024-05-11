import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "/gpu:0"
from keras.backend import clear_session
import pandas as pd
import re
import numpy as np
from keras import Sequential
from keras.layers import Dense, Dropout


def horizRead(file_path):
    a = {}
    HandleData = []
    data = pd.read_csv(file_path)
    SSseq = ""
    AAC = ""
    # id=file[0:-6]
    for i in range(len(data)):
        ls = data["# PSIPRED HFORMAT (PSIPRED V4.0)"][i]
        if ls[0:4] == "Pred":
            SSseq = SSseq + ls[6:]
        elif ls[0:4] == "  AA":
            AAC = AAC + ls[6:]

    a["AAC"] = AAC
    a["SSseq"] = SSseq

    HandleData.append(a)
    variables = list(HandleData[0].keys())
    dataframe = pd.DataFrame(
        [[i[j] for j in variables] for i in HandleData], columns=variables
    )
    return dataframe


def computeFeature(data):
    feature = []
    N = len(data)
    NH = data.count("H")
    NE = data.count("E")
    PH = NH / N
    PE = NE / N
    feature.append(PH)
    feature.append(PE)
    PHj = 0
    PEj = 0
    MAXSegH = 0
    MAXSegE = 0
    lsE = []
    lsH = []
    SS = ""
    AH = 0
    AE = 0
    for i in range(N):
        if data[i] == "H":
            endH = 0
            if i - 1 < 0 or data[i - 1] == "C" or data[i - 1] == "E":
                startH = i
            if i + 1 == N or data[i + 1] == "C" or data[i + 1] == "E":
                endH = i
                SS += "H"
            if endH != 0:
                AH += endH - startH
                tuple = (startH, endH)
                lsH.append(tuple)
            PHj += i
            NHN = 1
            for j in range(i + 1, N):
                if data[j] == "H":
                    NHN += 1
                else:
                    break
            if NHN > MAXSegH:
                MAXSegH = NHN
        elif data[i] == "E":
            endE = 0
            if i - 1 < 0 or data[i - 1] == "C" or data[i - 1] == "H":
                startE = i
            if i + 1 == N or data[i + 1] == "C" or data[i + 1] == "H":
                endE = i
                SS += "E"
            if endE != 0:
                AE += endE - startE
                tuple = (startE, endE)
                lsE.append(tuple)
            PEj += i
            NEN = 1
            for j in range(i + 1, N):
                if data[j] == "E":
                    NEN += 1
                else:
                    break
            if NEN > MAXSegE:
                MAXSegE = NEN
    CMVH = PHj / (N * (N - 1))
    CMVE = PEj / (N * (N - 1))
    MHN = MAXSegH / N
    MEN = MAXSegE / N
    if len(lsH) == 0:
        NAvgH = 0
    else:
        NAvgH = AH / len(lsH)

    if len(lsE) == 0:
        NAvgE = 0
    else:
        NAvgE = AE / len(lsE)
    feature.append(CMVH)
    feature.append(CMVE)
    feature.append(MHN)
    feature.append(MEN)
    feature.append(NAvgH)
    feature.append(NAvgE)
    lsP = []
    lsAP = []
    for i in range(len(lsE) - 1):
        start = lsE[i][1]
        end = lsE[i + 1][0]
        s = data[start:end]
        if s.count("H") != 0:
            lsP.append(i)
            lsP.append(i + 1)
        else:
            lsAP.append(i)
            lsAP.append(i + 1)
    PNE = len(set(lsP))
    APNE = len(set(lsAP))
    NPNE = PNE / N
    NAPNE = APNE / N
    feature.append(NPNE)
    feature.append(NAPNE)
    DHE = re.findall("HC*?E", data)
    DEH = re.findall("EC*?H", data)
    MaxDHE = 0
    MaxDEH = 0
    for i in DHE:
        if len(i) >= MaxDHE:
            MaxDHE = len(i) - 2
    for i in DEH:
        if len(i) >= MaxDEH:
            MaxDEH = len(i) - 2
    MaxDHEN = MaxDHE / N
    MaxDEHN = MaxDEH / N
    feature.append(MaxDHEN)
    feature.append(MaxDEHN)
    CountH6 = 0
    CountH8 = 0
    for i in lsH:
        if i[1] - i[0] > 5:
            CountH6 += 1
        if i[1] - i[0] > 7:
            CountH8 += 1
    if len(lsH) == 0:
        NCountH6 = 0
        NCountH8 = 0
    else:
        NCountH6 = CountH6 / len(lsH)
        NCountH8 = CountH8 / len(lsH)
    feature.append(NCountH6)
    feature.append(NCountH8)
    CountE5 = 0
    for i in lsE:
        if i[1] - i[0] > 4:
            CountE5 += 1
    if len(lsE) == 0:
        NCountE5 = 0
    else:
        NCountE5 = CountE5 / len(lsE)
    feature.append(NCountE5)

    SS = SS.replace("C", "")
    HES = data.replace("C", "")
    ISSS = data
    a = [
        "CHHE",
        "EHHC",
        "CHHC",
        "EHHE",
        "HEH",
        "HEC",
        "CEH",
        "CEC",
        "CHE",
        "EHC",
        "CHC",
        "EHE",
    ]
    for i in a:
        ISSS = ISSS.replace(i, i[0] + i[-1])
    IHES = ISSS.replace("C", "")
    N1 = len(HES)
    N11 = len(IHES)
    N111 = len(SS)
    if N1 == 0:
        THE = 0
        THH = 0
        TEE = 0
        TEH = 0
    else:
        THE = HES.count("HE") / N1
        THH = HES.count("HH") / N1
        TEE = HES.count("EE") / N1
        TEH = HES.count("EH") / N1
    if N11 == 0:
        ITHE = 0
        ITHH = 0
        ITEE = 0
        ITEH = 0
    else:
        ITHE = IHES.count("HE") / N11
        ITHH = IHES.count("HH") / N11
        ITEE = IHES.count("EE") / N11
        ITEH = IHES.count("EH") / N11
    feature.append(THE)
    feature.append(THH)
    feature.append(TEE)
    feature.append(TEH)
    feature.append(ITHE)
    feature.append(ITHH)
    feature.append(ITEE)
    feature.append(ITEH)
    PESj = 0
    PHSj = 0
    for i in range(len(SS)):
        if SS[i] == "H":
            PHSj += i
        if SS[i] == "E":
            PESj += i
    if N111 == 0:
        PHS = 0
        PES = 0
        CMVHS = 0
        CMVES = 0
    elif (N111 - 1) == 0:
        PHS = SS.count("H") / N111
        PES = SS.count("E") / N111
        CMVHS = 0
        CMVES = 0
    else:
        PHS = SS.count("H") / N111
        PES = SS.count("E") / N111
        CMVHS = PHSj / (N111 * (N111 - 1))
        CMVES = PESj / (N111 * (N111 - 1))
    feature.append(PHS)
    feature.append(PES)
    feature.append(CMVHS)
    feature.append(CMVES)
    return feature


def predictA(feature):
    clear_session()
    model = Sequential()
    model.add(Dense(100, input_shape=(27,), activation="selu"))
    model.add(Dense(200, activation="selu"))
    model.add(Dropout(0.2))
    model.add(Dense(4, activation="softmax"))
    model.load_weights("scop_classifier_model.hdf5")
    model.compile(
        loss="sparse_categorical_crossentropy", optimizer="Adam", metrics=["acc"]
    )
    # with graph.as_default():
    #     y_pred = model.predict(feature)

    y_pred = model.predict(feature)
    y_pred = y_pred.argmax(axis=1)
    if y_pred == 0:
        result = "This protein belongs to the α class!"
    elif y_pred == 1:
        result = "This protein belongs to the β class!"
    elif y_pred == 2:
        result = "This protein belongs to the α/β class!"
    elif y_pred == 3:
        result = "This protein belongs to the α+β class!"
    else:
        result = "Its secondary structure has not been predicted"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horiz_file_path", type=str)
    args = parser.parse_args()
    SS = horizRead(args.horiz_file_path)
    features = []
    features.append(computeFeature(SS["SSseq"][0]))
    features = np.array(features)
    classF = predictA(features)
    print(classF)
