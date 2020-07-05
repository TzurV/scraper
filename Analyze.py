import pandas as pd
import glob, os


if __name__ == "__main__":

    print("#=====================================================")
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
    print(allFundsInf.columns)
    #print(allFundsInf.head())
    #print(allFundsInf.columns)
    #print(allFundsInf.head())

    #=============================
    #                      3 first rows and 5 columns
    #print(allFundsInf.iloc[:3, 0:5])
    #   Unnamed: 0            date                                       fundName  Quartile  FERisk
    #0           0  14/06/20 23:33  HSBC World Selection Cautious Portfolio C Acc       NaN    27.0
    #1           1  14/06/20 23:33                         Jupiter European I Acc       2.0    79.0
    #2           2  14/06/20 23:33                   Liontrust Global Alpha C Acc       1.0    89.0

    #==========================
    # Points (like histogram)
    #print(allFundsInf.groupby('fundName').fundName.count())
    #fundsList = allFundsInf.groupby('fundName')
    #print(type(fundsList))

    # ===========================
    # https://www.shanelynn.ie/summarising-aggregation-and-grouping-data-in-python-pandas/
    print(allFundsInf.groupby('Sector').Sector.count())
    print(allFundsInf.groupby(['fundName', 'date'])['date'].count())
    #AllSectors = allFundsInf.groupby('Sector')

    #https://realpython.com/pandas-groupby/
    #groupedFundsList = allFundsInf.groupby(['Sector','fundName', 'date'], as_index=False)
    #print(groupedFundsList)
    #for (sector, fund, date, frame) in groupedFundsList:
    #    print(sector, fund, date)

    # create report
    COLUMN_NAMES=['fundName', 'worsenQuartile', 'worsenFERisk']
    pdSummary = pd.DataFrame(columns=COLUMN_NAMES)

    # group by fundname
    groupedFundsList = allFundsInf.groupby('fundName', as_index=False)
    #print(groupedFundsList)
    for fund, frame in groupedFundsList:
        print(fund)
        #print(frame.sort_values(by="date"))
        sortedFunds = frame.sort_values(by="date")
        print(sortedFunds[['date', 'Quartile', 'FERisk', '3m', '6m', '1y', '3y', '5y' ]])
        
        # check if Quartile and FERisk is different
        lastQuartile = sortedFunds.Quartile.iloc[0]
        worsenQuartile = False
        lastFERisk = sortedFunds.FERisk.iloc[0]
        worsenFERisk = False
        for i in range(1, len(sortedFunds.index)):
            if(lastQuartile>sortedFunds.Quartile.iloc[i]):
                worsenQuartile = True
            if(lastFERisk*0.95>sortedFunds.FERisk.iloc[i]):
                worsenFERisk = True
        if worsenQuartile or worsenFERisk:
            print("\t#==================== Check this one =================")
            print(f"\t# {worsenQuartile}: worsen Quartile, higher FERisk: {worsenFERisk}  \n")
            pdSummary = pdSummary.append({'fundName':fund, 'worsenQuartile':worsenQuartile, 'worsenFERisk':worsenFERisk}, ignore_index=True)
            

    print("====== Observations summary ============")
    print(pdSummary)

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



        



