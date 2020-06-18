import pandas as pd
import glob, os


if __name__ == "__main__":

    allFundsInf = pd.DataFrame()
    for file in glob.glob("*_FundsInf.csv"):
        print(f"Loading {file}" )
        fundInf = pd.read_csv(file, sep=',')
        allFundsInf = allFundsInf.append(fundInf, ignore_index=True)
        #print(fundInf.head())
        #print(fundInf.shape)
        #print(fundInf.columns)

print("-------- ALL ----------------")
print(allFundsInf.shape)
#print(allFundsInf.head())
#print(allFundsInf.columns)
print(allFundsInf.head())


print(allFundsInf.iloc[:3, 0:5])

# Points (like histogram)
print(allFundsInf.groupby('fundName').fundName.count())
print(allFundsInf.groupby('Sector').Sector.count())
AllSectors =allFundsInf.groupby('Sector').Sector.count()
#print(type(A), A.shape)
AllSectorsdict = AllSectors.to_dict()
for sector in AllSectorsdict:
    print(f"A[{sector}]={AllSectorsdict[sector]}")

    # Conditional Selection
    FundsInSector = allFundsInf.loc[(allFundsInf.Sector == sector)]
    print(FundsInSector)



    



