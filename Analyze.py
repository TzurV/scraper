import pandas as pd
import glob, os
import pprint
from datetime import datetime
import sys

if __name__ == "__main__":

    #================================
    # load current holdings from Holdings.txt
    holdingsList = {}
    HoldingsFile = "Holdings.txt"
    print("# Loading current holdings funds list:")
    with open(HoldingsFile) as fp:
        line = fp.readline()

        while line:
            print("Fund |{}|".format(line.strip()))
            holdingsList[line.strip()] = False
            line = fp.readline()

    print("#=====================================================")
    # create empty dataframe
    allFundsInf = pd.DataFrame()
    for file in glob.glob("*_FundsInf.csv"):
        print(f"Loading {file}" )

        # loading examples
        #https://chrisalbon.com/python/data_wrangling/pandas_dataframe_importing_csv/

        fundInf = pd.read_csv(file, sep=',')

        dtm = lambda x: datetime.strptime(x, "%d/%m/%y %H:%M")
        fundInf["date"] = fundInf["date"].apply(dtm)

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
    
    # conver date column to a datetime object
    #allFundsInf['date'] = pd.to_datetime(allFundsInf['date']).dt.strftime("%m/%d/%y")
    #print(type(allFundsInf['date']))
    
    #dtm = lambda x: datetime.strptime(x, "%d/%m/%y %H:%M")
    #allFundsInf["date"] = allFundsInf["date"].apply(dtm)
    
    #dtm1 = lambda x: datetime.date()
    #allFundsInf["date"] = allFundsInf["date"].datetime.date
    
    
    # get unique fund code
    fundCode = lambda P: P.split('/')[-2]
    allFundsInf["fundCode"] = allFundsInf["url"].apply(fundCode)


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
    COLUMN_NAMES=['fundName', 'Holding', 'worsenQuartile', 'worsenFERisk']
    pdSummary = pd.DataFrame(columns=COLUMN_NAMES)

    # group by fundname
    #groupedFundsList = allFundsInf.groupby('fundName', as_index=False)
    
    # group by fundname (code)
    groupedFundsList = allFundsInf.groupby('fundCode', as_index=False)
    
    #print(groupedFundsList)
    for fundCode, frame in groupedFundsList:


        #print(frame.sort_values(by="date"))
        sortedFunds = frame.drop_duplicates(subset ="date", keep = False)
        #sortedFunds = frame1.sort_values(by="date", ascending=True)
        
        #sortedFunds.drop_duplicates(subset ="date", keep = False, inplace = True)

        # date  column becomes index column
        sortedFunds.set_index('date', drop=False, append=False, inplace=True, verify_integrity=False)
        sortedFunds.sort_index()

        # reverse order
        sortedFunds = sortedFunds.iloc[::-1]

        # get fund name
        fund = sortedFunds.fundName.iloc[0]

        Holding = (fund in holdingsList)
        print(f"\n-=> Fund {fund} , holding {Holding} <=-")
        print(f"\t\tCode:{fundCode}")
        if Holding:
            holdingsList[fund] = True
        print(sortedFunds[['Quartile', 'FERisk', '3m', '6m', '1y', '3y', '5y' ]])
        

        #print(sortedFunds[['date', 'Quartile', 'FERisk', '3m', '6m', '1y', '3y', '5y' ]])
        
        # check if Quartile and FERisk is different
        lastQuartile = sortedFunds.Quartile.iloc[0]
        worsenQuartile = False
        lastFERisk = sortedFunds.FERisk.iloc[0]
        worsenFERisk = False
        for i in range(1, len(sortedFunds.index)):
            #print(sortedFunds.iloc[[i]])
            if(lastQuartile>sortedFunds.Quartile.iloc[i]):
                worsenQuartile = True
            if(lastFERisk*0.95>sortedFunds.FERisk.iloc[i]):
                worsenFERisk = True
        if worsenQuartile or worsenFERisk:
            print("\t#==================== Check this one =================")
            print(f"\t# {worsenQuartile}: worsen Quartile, higher FERisk: {worsenFERisk}  \n")
            pdSummary = pdSummary.append({'fundName':fund, 
                                          'Holding':Holding ,
                                          'worsenQuartile':worsenQuartile, 
                                          'worsenFERisk':worsenFERisk}, ignore_index=True)
            

    print("----------- Check if all holdings are monitored --------")
    pprint.pprint(holdingsList)
    for a in holdingsList:
        if not holdingsList[a]:
            print(f"Check fund {a}")


    print("")
    print("====== Observations summary ============")
    print(pdSummary.to_string())

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



        



