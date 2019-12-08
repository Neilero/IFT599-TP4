import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm

BASE_PATH = "MovieLens/100K/ml-100k/u"
HEADER = ["user id", "item id", "rating", "timestamp"]

def importData(dataIndex):
    trainDataPath = "{}{}.base".format(BASE_PATH, dataIndex)
    testDataPath = "{}{}.test".format(BASE_PATH, dataIndex)

    baseData = pd.read_csv(trainDataPath, sep='\t', names=HEADER)
    testData = pd.read_csv(testDataPath, sep='\t', names=HEADER)

    return baseData, testData

def pearsonCorrelation(userId1, userId2):
    global baseData
    global correlationMatrix

    if correlationMatrix[userId1, userId2] == 0:
        user1CommonRatings, user2CommonRatings = getCommonUsersRattings(userId1, userId2)

        # If user 1 and 2 have less than 2 common ratings, the correlation coefficient is not defined
        if len(user1CommonRatings) < 2:
            return np.nan
        # If the ratings are constant, the correlation coefficient is not defined
        if (user1CommonRatings == user1CommonRatings[0]).all() or (user2CommonRatings == user2CommonRatings[0]).all():
            return np.nan

        pc, _ = pearsonr(user2CommonRatings, user1CommonRatings)

        correlationMatrix[userId1, userId2] = pc
        correlationMatrix[userId2, userId1] = pc    # because the correlation is symmetric

    return correlationMatrix[userId1, userId2]

# def getPearsonCorrelationMatrix(data):
#     userList = np.unique(data["user id"])
#     correlationMatrix = np.zeros((userList.max()+1, userList.max()+1))
#
#     start = time.time()
#     with ThreadPoolExecutor() as executor:
#         for user1, user2, correlation in executor.map(lambda args : pearsonCorrelation(*args), product(userList, userList, [data])):
#             correlationMatrix[user1][user2] = correlation
#     finish = time.time()
#
#     print("Pearson correlation computing : {:.3f}".format(finish-start))
#
#     return correlationMatrix

def getCommonUsersRattings(userId1, userId2):
    global baseData

    # Filter according to user's id
    user1 = baseData[ baseData["user id"] == userId1 ]
    user2 = baseData[ baseData["user id"] == userId2 ]

    # Inner join users on items
    commonRatings = user1.merge(user2, on="item id", suffixes=(' u1', ' u2'))

    return commonRatings["rating u1"], commonRatings["rating u2"]

def getRatings(userId, itemId, data):
    ratingRowFilter = (data["user id"] == userId) & (data["item id"] == itemId)
    ratingRow = data[ratingRowFilter]
    return ratingRow["rating"].values

def getUserNeighbors(userId, k):
    global baseData
    global userNeighbor

    if isinstance(userNeighbor[userId], float):
        userList = np.unique(baseData["user id"])
        userList = np.delete(userList, np.where(userList==userId))

        # associate other users with their similarity to the given user
        userList = np.array([[otherUserId, pearsonCorrelation(userId, otherUserId)] for otherUserId in userList])

        # remove users with non existing similarity (i.e. when similarity is np.NaN)
        userList = userList[~np.isnan(userList).any(axis=1)]

        # sort by similarity and return the first k values
        sortedIndexes = userList[:,1].argsort()[::-1]

        # store neighbor
        userNeighbor[userId] = userList[sortedIndexes]

    return userNeighbor[userId][:k]

def predictRating(userId, itemId, k=5):
    global baseData

    userNeighbors = getUserNeighbors(userId, k)

    prediction = 0
    similaritySum = 0
    for neighbor, neighborSimilarity in userNeighbors:
        neighborRatings = getRatings(neighbor, itemId, baseData)
        prediction += neighborRatings.sum()
        similaritySum += abs(neighborSimilarity) * len(neighborRatings)

    if similaritySum > 0:
        return prediction / similaritySum
    return np.NaN

def RMSEerror(ratingRow):

    prediction = predictRating(ratingRow["user id"], ratingRow["item id"])
    if np.isnan(prediction):
        return 0    # don't count unpredictable ratings

    rowError = (prediction - ratingRow["rating"]) ** 2

    return rowError

def RMSE(baseData, testData):
    global correlationMatrix
    global userNeighbor

    userNumber = baseData["user id"].max() + 1
    correlationMatrix = np.zeros((userNumber, userNumber))
    userNeighbor = np.zeros(userNumber).tolist()

    globalError = 0
    for i, row in tqdm(testData.iterrows(), total=len(testData)):
        globalError += RMSEerror(row)

    globalError = np.sqrt( globalError / len(testData) )

    return globalError

def main():
    global baseData

    for dataIndex in range(1, 6):
        print("--- Testing algorithm on data {} ---".format(dataIndex))

        baseData, testData = importData(dataIndex)
        error = RMSE(baseData, testData)

        print("Validation error : {:.3f} %\n".format(error))

if __name__ == '__main__':
    main()