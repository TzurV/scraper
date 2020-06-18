import pandas as pd
import glob, os


if __name__ == "__main__":

    allFundsInf = pd.DataFrame()
    for file in glob.glob("*_FundsInf.csv"):
        print(file)
        fundInf = pd.read_csv(file, sep=',')
        allFundsInf = allFundsInf.append(fundInf, ignore_index=True)
        print(fundInf.head())
        print(fundInf.shape)
        print(fundInf.columns)

print("-------- ALL ----------------")
print(allFundsInf.shape)
print(allFundsInf.head())
print(allFundsInf.columns)
print(allFundsInf.head())


print(allFundsInf.iloc[:, 0:5])
