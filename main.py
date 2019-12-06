import pandas as pd
from scipy.stats import pearsonr

BASE_PATH = "MovieLens/100K/ml-100k/u"
HEADER = ["user id", "item id", "rating", "timestamp"]

def importData(dataIndex):
    trainDataPath = BASE_PATH + dataIndex + ".base"
    testDataPath = BASE_PATH + dataIndex + ".test"

    trainData = pd.read_csv(trainDataPath, sep='\t', names=HEADER)
    testData = pd.read_csv(testDataPath, sep='\t', names=HEADER)

    return trainData, testData

def pearsonCorrelation(userID1, userID2, data):
    user1CommonRatings, user2CommonRatings = getCommonUsersRattings(userID1, userID2, data)

    pc, _ = pearsonr(user2CommonRatings, user1CommonRatings)

    return pc

def getCommonUsersRattings(userID1, userID2, data):
    # Filter according to user's id
    user1 = data[ data["user id"] == userID1 ]
    user2 = data[ data["user id"] == userID2 ]

    # Inner join users on items
    commonRatings = user1.merge(user2, on="item id", suffixes=(' u1', ' u2'))

    return commonRatings["rating u1"], commonRatings["rating u2"]

def main():
    print("Hello world!")

if __name__ == '__main__':
    main()