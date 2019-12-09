import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm

BASE_PATH = "MovieLens/100K/ml-100k/u"
HEADER = ["user id", "item id", "rating", "timestamp"]

def importData(dataIndex):
    """
    Import base and test data from the uX.base and uX.test files, with X being the given number.
    :param dataIndex: index of the files to import (must be between 1 and 5 included)
    :return: base and test pandas' dataFrames
    """
    trainDataPath = "{}{}.base".format(BASE_PATH, dataIndex)
    testDataPath = "{}{}.test".format(BASE_PATH, dataIndex)

    baseData = pd.read_csv(trainDataPath, sep='\t', names=HEADER)
    testData = pd.read_csv(testDataPath, sep='\t', names=HEADER)

    return baseData, testData

def pearsonCorrelation(userId1, userId2, baseData, correlationMatrix):
    """
    Compute the pearson correlation if not already done for the given users.
    The result is stocked to avoid computing it again.
    :param userId1: the first user's id
    :param userId2: the second user's id
    :param baseData: the data where to compute the correlation
    :param correlationMatrix: the matrix where the computed correlations are stored
    :return: the pearson correlation of the given users
    """
    if correlationMatrix[userId1, userId2] == 0:
        user1CommonRatings, user2CommonRatings = getCommonUsersRatings(userId1, userId2, baseData)

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

# This method is an attempt to parallelize the pre-processing of the correlation matrix
# It has been canceled because it was taking too much time to compute and because the parallelism didn't worked
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

def getCommonUsersRatings(userId1, userId2, baseData):
    """
    Returns the common ratings between the given users
    :param userId1: the first user's id
    :param userId2: the second user's id
    :return: a list of the common ratings between the given users
    """
    # Filter according to user's id
    user1 = baseData[ baseData["user id"] == userId1 ]
    user2 = baseData[ baseData["user id"] == userId2 ]

    # Inner join users on items
    commonRatings = user1.merge(user2, on="item id", suffixes=(' u1', ' u2'))

    return commonRatings["rating u1"], commonRatings["rating u2"]

def getRatings(userId, itemId, data):
    """
    returns the ratings given to the given item by the given user in the given data
    :param userId: the user's id
    :param itemId: the item's id
    :param data: the data where to look for the ratings
    :return: a list of the ratings
    """
    ratingRowFilter = (data["user id"] == userId) & (data["item id"] == itemId)
    ratingRow = data[ratingRowFilter]
    return ratingRow["rating"].values

def getUserNeighbors(userId, k, baseData, correlationMatrix, userNeighbor):
    """
    Returns the k nearest neighbors' id of the given user using pearson's correlation
    The results is store in the given last attribute to avoid computing it again
    :param userId: the user's id
    :param k: the number of wanted neighbors
    :param baseData: the data where to look for
    :param correlationMatrix: the matrix where the computed correlations are stored
    :param userNeighbor: the list where the neighbors are stored once computed
    :return: a list of the k nearest neighbors' id of the given user
    """
    if isinstance(userNeighbor[userId], float):
        userList = np.unique(baseData["user id"])
        userList = np.delete(userList, np.where(userList==userId))

        # associate other users with their similarity to the given user
        userList = np.array([[otherUserId, pearsonCorrelation(userId, otherUserId, baseData, correlationMatrix)] for otherUserId in userList])

        # remove users with non existing similarity (i.e. when similarity is np.NaN)
        userList = userList[~np.isnan(userList).any(axis=1)]

        # sort by similarity and return the first k values
        sortedIndexes = userList[:,1].argsort()[::-1]

        # store neighbor
        userNeighbor[userId] = userList[sortedIndexes]

    k = min(k, len(userNeighbor[userId]))

    return userNeighbor[userId][:k]

def predictRating(userId, itemId, baseData, correlationMatrix, userNeighbor, k=10):
    """
    Predict the rating for the given user on the given item
    We take into account the possibility for a user to have rated an item several times
    :param userId: the user's id
    :param itemId: the item's id
    :param baseData: the data to use for the prediction
    :param correlationMatrix: the correlation matrix where the correlations are stored
    :param userNeighbor: the list of stored user neighbors
    :param k: the number of neighbor to use for the prediction
    :return: the predicted rating for given user on the given item
    """
    userNeighbors = getUserNeighbors(userId, k, baseData, correlationMatrix, userNeighbor)

    prediction = 0
    similaritySum = 0
    for neighbor, neighborSimilarity in userNeighbors:
        neighborRatings = getRatings(neighbor, itemId, baseData)
        prediction += neighborSimilarity * neighborRatings.sum()
        similaritySum += abs(neighborSimilarity) * len(neighborRatings)

    if similaritySum > 0:
        return prediction / similaritySum
    return np.NaN

def RMSEerror(ratingRow, k, baseData, correlationMatrix, userNeighbor):
    """
    Compute the error for one test row
    :param ratingRow: the test row from which to compute the error
    :param k: the number of neighbor to take into account
    :param baseData: the base data to use for the prediction
    :param correlationMatrix: where the correlations are stored and saved (for optimization purpose)
    :param userNeighbor: where the neighbors are stored and saved (for optimization purpose)
    :return: the error corresponding to the given row
    """
    prediction = predictRating(ratingRow["user id"], ratingRow["item id"], baseData, correlationMatrix, userNeighbor, k)
    if np.isnan(prediction):
        return 0    # don't count unpredictable ratings

    rowError = (prediction - ratingRow["rating"]) ** 2

    return rowError

def RMSE(baseData, testData, k=10):
    """
    Compute the RMSE for the given base data using the given test data.
    :param baseData: the data to use for the predictions
    :param testData: the data to use for the tests
    :param k: the number of neighbor to take into account
    :return: the global RMSE of the given data
    """
    userNumber = baseData["user id"].max() + 1
    correlationMatrix = np.zeros((userNumber, userNumber))
    userNeighbor = np.zeros(userNumber).tolist()

    globalError = 0
    for i, row in tqdm(testData.iterrows(), total=len(testData)):
        globalError += RMSEerror(row, k, baseData, correlationMatrix, userNeighbor)

    globalError = np.sqrt( globalError / len(testData) )

    return globalError

def main():
    """
    The main testing program which prints the results
    :return: None
    """
    for k in range(3, 17, 4):
        for dataIndex in range(1, 6):
            print("--- Testing algorithm on data {} (with k = {}) ---".format(dataIndex, k))

            baseData, testData = importData(dataIndex)
            error = RMSE(baseData, testData, k)

            print("Validation error : {:.3f} %\n".format(error*100))

if __name__ == '__main__':
    main()