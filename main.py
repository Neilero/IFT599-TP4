import pandas as pd
import numpy as np
from scipy.stats import pearsonr

BASE_PATH = "MovieLens/100K/ml-100k/u"
HEADER = ["user id", "item id", "rating", "timestamp"]

def importData(dataIndex):
    trainDataPath = "{}{}.base".format(BASE_PATH, dataIndex)
    testDataPath = "{}{}.test".format(BASE_PATH, dataIndex)

    baseData = pd.read_csv(trainDataPath, sep='\t', names=HEADER)
    testData = pd.read_csv(testDataPath, sep='\t', names=HEADER)

    return baseData, testData

def pearsonCorrelation(userId1, userId2, data):
    user1CommonRatings, user2CommonRatings = getCommonUsersRattings(userId1, userId2, data)

    pc, _ = pearsonr(user2CommonRatings, user1CommonRatings)

    return pc

def getCommonUsersRattings(userId1, userId2, data):
    # Filter according to user's id
    user1 = data[ data["user id"] == userId1 ]
    user2 = data[ data["user id"] == userId2 ]

    # Inner join users on items
    commonRatings = user1.merge(user2, on="item id", suffixes=(' u1', ' u2'))

    return commonRatings["rating u1"], commonRatings["rating u2"]

def getRatings(userId, itemId, data):
    ratingRowFilter = (data["user id"] == userId) & (data["item id"] == itemId)
    ratingRow = data[ratingRowFilter]
    return ratingRow["rating"].values

def predictRating(userId, itemId, data, k=5):
    userNeighbors = getUserNeighbors(userId, k, data)

    prediction = 0
    similaritySum = 0
    for neighbor in userNeighbors:
        neighborRatings, neighborSimilarity = getRatings(neighbor, itemId, data)
        prediction += neighborRatings.sum()
        similaritySum += abs(neighborSimilarity) * len(neighborRatings)

    if similaritySum > 0:
        return prediction / similaritySum
    return np.NaN

def RMSE(baseData, testData):
    error = 0

    # for each test row : error += (predict(u, i) - r)**2
    for i, row in testData.iterrows():
        rowError = (predictRating(row["user id"], row["item id"], baseData) - row["rating"])**2
        # error += np.nan_to_num(rowError)
        error += rowError

    error = np.sqrt( error/len(testData) )

    return error

def main():
    for dataIndex in range(6):
        print("--- Computing data {} error ---".format(dataIndex))

        baseData, testData = importData(dataIndex)
        error = RMSE(baseData, testData)

        print("Validation error : {:.3f} %\n".format(error))

if __name__ == '__main__':
    main()