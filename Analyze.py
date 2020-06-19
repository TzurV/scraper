import pandas as pd
import glob, os


if __name__ == "__main__":

    # create empty dataframe
    allFundsInf = pd.DataFrame()
    for file in glob.glob("*_FundsInf.csv"):
        print(f"Loading {file}" )

        # loading examples
        #https://chrisalbon.com/python/data_wrangling/pandas_dataframe_importing_csv/

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

    #                      3 first rows and 5 columns
    print(allFundsInf.iloc[:3, 0:5])
    #   Unnamed: 0            date                                       fundName  Quartile  FERisk
    #0           0  14/06/20 23:33  HSBC World Selection Cautious Portfolio C Acc       NaN    27.0
    #1           1  14/06/20 23:33                         Jupiter European I Acc       2.0    79.0
    #2           2  14/06/20 23:33                   Liontrust Global Alpha C Acc       1.0    89.0

    # Points (like histogram)
    print(allFundsInf.groupby('fundName').fundName.count())
    fundsList = allFundsInf.groupby('fundName')
    print(type(fundsList))

    print(allFundsInf.groupby('Sector').Sector.count())
    AllSectors = allFundsInf.groupby('Sector')

    #https://realpython.com/pandas-groupby/
    #groupedFundsList = allFundsInf.groupby(['Sector','fundName', 'date'], as_index=False)
    #print(groupedFundsList)
    #for (sector, fund, date, frame) in groupedFundsList:
    #    print(sector, fund, date)

    groupedFundsList = allFundsInf.groupby('fundName', as_index=False)
    print(groupedFundsList)
    for fund, frame in groupedFundsList:
        print(fund)
        print(frame)



    #print(type(A), A.shape)
    #AllSectorsdict = AllSectors.to_dict()
    #for sector in AllSectorsdict:
    #    print(f"A[{sector}]={AllSectorsdict[sector]}")
    #
    #    # Conditional Selection
    #    FundsInSector = allFundsInf.loc[(allFundsInf.Sector == sector)]
    #    print(FundsInSector.shape)
    #    for r in range(len(FundsInSector.index)):
    #        print(FundsInSector.iloc[r,:])
    #    #for fund in FundsInSector:
    #    #print(type(FundsInSector))



        



