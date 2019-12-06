import pandas as pd
from scipy.stats import pearsonr

BASE_PATH = "MovieLens/100K/ml-100k/u"
HEADER = ["user id", "item id", "rating", "timestamp"]

def importData(dataIndex):
    baseDataPath = BASE_PATH + dataIndex + ".base"
    testDataPath = BASE_PATH + dataIndex + ".test"

    baseData = pd.read_csv(baseDataPath, sep='\t', names=HEADER)
    testData = pd.read_csv(testDataPath, sep='\t', names=HEADER)

def main():
    print("Hello world!")

if __name__ == '__main__':
    main()