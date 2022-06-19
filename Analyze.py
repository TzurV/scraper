import pandas as pd
import glob, os
import pprint
from datetime import datetime
import sys

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":

    #================================
    # load current holdings from Holdings.txt
    # holdingsList = {}
    # HoldingsFile = "Holdings.txt"
    # print("# Loading current holdings funds list:")
    # with open(HoldingsFile) as fp:
    #     line = fp.readline()

    #     while line:
    #         print("Fund |{}|".format(line.strip()))
    #         holdingsList[line.strip()] = False
    #         line = fp.readline()

    print("#=====================================================")
    print("-------- loading '*_FundsInf.csv'  ----------------")

    # create empty dataframe
    allFundsInf = pd.DataFrame()
    loadedFile = 0
    last_file = None
    last_fundInf = None
    print(f"current working directory {os. getcwd()}")
    for file in glob.glob("*_FundsInf.csv"):
        # print(f"Loading {file}" )
        last_file = file
        loadedFile +=1

        # loading examples
        #https://chrisalbon.com/python/data_wrangling/pandas_dataframe_importing_csv/

        fundInf = pd.read_csv(file, sep=',')

        dtm = lambda x: datetime.strptime(x, "%d/%m/%y %H:%M")
        fundInf["date"] = fundInf["date"].apply(dtm)

        # get unique fund code
        fundCode = lambda P: P.split('/')[-2]
        fundInf["fundCode"] = fundInf["url"].apply(fundCode)
        last_fundInf = fundInf

        allFundsInf = allFundsInf.append(fundInf, ignore_index=True)
        #print(fundInf.head())
        #print(fundInf.shape)
        #print(fundInf.columns)

    print(f"# Total '*_FundsInf.csv' files loaded {loadedFile}")

    # get unique fund code
    # fundCode = lambda P: P.split('/')[-2]
    # allFundsInf["fundCode"] = allFundsInf["url"].apply(fundCode)

    print(allFundsInf.columns)

    # update holdingsList based on the last in the list 
    # (should be the latest one)
    print(f"# gathering holding information from {last_file}")
    holdingsList = {}
    activeMonitoredList = []
    for fund in last_fundInf.itertuples():
        activeMonitoredList.append(fund.fundCode)
        if fund.Hold:
            # will set to true when the fund is analyzed
            holdingsList[fund.fundName] = False
              
    print("-------- loaded completed ----------------")
 

    print("# Load Sector Information =====================================================")
    # create empty dataframe
    sectorSelectedColumns = ['1m', '3m', '6m', '1y', '3y', '5y']
    allSectorsInf = pd.DataFrame()
    totalSectorsInfFiles = 0
    last_sectorsInf = None
    for file in glob.glob("*_TrustNetSectors.csv"):
        #print(f"Loading {file}" )
        totalSectorsInfFiles += 1

        sectorsInf = pd.read_csv(file, sep=',', dayfirst=True)
        dtm = lambda x: datetime.strptime(x, "%d/%m/%y %H:%M")
        sectorsInf["date"] = sectorsInf["date"].apply(dtm)
        
        # removes rows that dont include numbers 
        #like  '21,20/09/20 09:52,IA Not yet assigned,-,-,-,-,-,-'
        sectorsInf = sectorsInf.drop(sectorsInf[sectorsInf['1m'] == '-'].index)
        for col in sectorSelectedColumns:
            sectorsInf[col] = sectorsInf[col].astype(float) 
        last_sectorsInf = sectorsInf
                
        allSectorsInf = allSectorsInf.append(sectorsInf, ignore_index=True)

    print("-------- ALL ----------------")
    print(f"# Total '*_TrustNetSectors.csv' files loaded {totalSectorsInfFiles}")
    print(f"Latest file is {last_file}")
    print(f"cout funds per sector over the whole data collection period {allSectorsInf.shape}")
    # print(allSectorsInf.columns)
    

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

    #print(f"\n{'*'*50}")    
    # ===========================
    # https://www.shanelynn.ie/summarising-aggregation-and-grouping-data-in-python-pandas/
    #print(allFundsInf.groupby('Sector').Sector.count())
    # print for each fund what dates data was collected
    #print(allFundsInf.groupby(['fundName', 'date'])['date'].count())

    print(f"\n{'-'*50}")  
    print('Current Number of funds per sector in the tracking list:')
    #print(last_fundInf.groupby(['Sector'])['Sector'].count())
    #AllSectors = allFundsInf.groupby('Sector')

    #print("\n# Sectors not represented in the funds tracking list. !")
    #represented_sectors = [a[0] for a in last_fundInf.groupby(['Sector'])]
    #for sector in list(last_sectorsInf['sectorName']):
    #    if sector not in represented_sectors:
    #        print(f"{sector}")
        
    sectors_summary = dict()
    for sector in list(last_sectorsInf['sectorName']):
        sectors_summary[sector]= {'count':'None', 'Weekly':'N/A'}        
    for a in last_fundInf.groupby(['Sector'])['Sector']:
        if a[0] not in sectors_summary:
            sectors_summary[a[0]]= {'count':'None', 'Weekly':'N/A'}              
        sectors_summary[a[0]]['count'] = len(a[1])
    for a in allFundsInf.groupby('Sector'):
        if a[0] not in sectors_summary:
            sectors_summary[a[0]]= {'count':'None', 'Weekly':'N/A'}            
        sectors_summary[a[0]]['Weekly'] = len(a[1])
    
    print(f"{'sector':<40}: {'funds':>4} {'Cases':>6} {sectorSelectedColumns}")
    for sector in sectors_summary:
        indx = last_sectorsInf.index[last_sectorsInf['sectorName']==sector]
        if len(indx)>0:
            print(f"{sector:<40}: {sectors_summary[sector]['count']:>4} {sectors_summary[sector]['Weekly']:>6} [{last_sectorsInf.loc[indx, sectorSelectedColumns].to_string(index=False, header=False, col_space=7)}]")
        else:
            print(f"{sector:<40}: {sectors_summary[sector]['count']:>4} {sectors_summary[sector]['Weekly']:>6}")

    #https://realpython.com/pandas-groupby/
    #groupedFundsList = allFundsInf.groupby(['Sector','fundName', 'date'], as_index=False)
    #print(groupedFundsList)
    #for (sector, fund, date, frame) in groupedFundsList:
    #    print(sector, fund, date)
    
    # create report
    COLUMN_NAMES=['fundName', 'Holding', 'from_date', '#weeks', 'past_held', 'worsenQuartile', 'worsenFERisk', 'worse3mThanSector', '3m', '6m', '1y']
    pdSummary = pd.DataFrame(columns=COLUMN_NAMES)

    # group by fundname
    #groupedFundsList = allFundsInf.groupby('fundName', as_index=False)
    
    # group by fundname (code)
    groupedFundsList = allFundsInf.groupby('fundCode', as_index=False)
    
    #print(groupedFundsList)
    for fundCode, frame in groupedFundsList:

        # skip funds not on the current excel list
        if not fundCode in activeMonitoredList:
            continue

        sortedFunds = frame.drop_duplicates(subset ="date", keep = False)

        # date  column becomes index column
        sortedFunds.set_index('date', drop=False, append=False, inplace=True, verify_integrity=False)
        sortedFunds.sort_index()

        # reverse order
        sortedFunds = sortedFunds.iloc[::-1]

        # get fund name
        fund = sortedFunds.fundName.iloc[0]

        # get Fund sector
        fundSector = sortedFunds['Sector'][0]

        latestDate = sortedFunds.date.iloc[0]
        print(f"Latest date {latestDate}")

        Holding = (fund in holdingsList)
        print(f"\n-=> Fund {fund} , holding {Holding} <=-")
        print(f"\t\tCode:{fundCode}\tSector: {fundSector}")
        if Holding:
            holdingsList[fund] = True

        worse3MthanSector = False
        fundSectorPerformanceOnDate = allSectorsInf.loc[ (allSectorsInf['date'].dt.date == \
                                   pd.to_datetime(latestDate, format='%Y-%m-%d', errors='ignore')) & \
                                       (allSectorsInf['sectorName'] == fundSector ) ]
        if len(fundSectorPerformanceOnDate):
            #print(fundSectorPerformanceOnDate[sectorSelectedColumns])
            sector3m = float(fundSectorPerformanceOnDate[sectorSelectedColumns]['3m'])
            fund3m = float(sortedFunds.iloc[0]['3m'])
            worse3MthanSector = bool(fund3m<sector3m)
        
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

        # count number of weeks fund is held
        #weeks_held = 0
        hold_inf = sortedFunds['Hold'].value_counts()
        weeks_held = 0
        from_date = None
        past_held = False
        if True in hold_inf.index:
            for i in range(0, len(sortedFunds.index)):
                if sortedFunds.Hold.iloc[i]:
                    from_date = sortedFunds.index[i].date()
                    weeks_held += 1
                else:
                    break
                    
            past_held = weeks_held>hold_inf[True]

        if from_date is not None:
            print(f"# Held since {from_date} for ~{weeks_held} weeks")
        print(sortedFunds[:20][['Quartile', 'FERisk', '3m', '6m', '1y', '3y', '5y', 'Hold' ]])
        

        # print fund analysis summary 
        if worsenQuartile or worsenFERisk or worse3MthanSector:
            print("\t#==================== Check this one =================")
            print(f"\t# {worsenQuartile}: worsen Quartile, higher FERisk: {worsenFERisk}, worst compare to Sector 3m: {worse3MthanSector} ")
            if worse3MthanSector:
                print(f"\t\tSector {sector3m} Fund {fund3m} -> {worse3MthanSector}")
            print("==\n")
            pdSummary = pdSummary.append({'fundName':fund, 
                                          'Holding':Holding,
                                          'from_date': from_date,
                                          '#weeks': weeks_held,
                                          'past_held': past_held,
                                          'worsenQuartile':worsenQuartile, 
                                          'worsenFERisk':worsenFERisk,
                                          'worse3mThanSector':worse3MthanSector,
                                          '3m':sortedFunds.iloc[0]['3m'],
                                          '6m':sortedFunds.iloc[0]['6m'], 
                                          '1y':sortedFunds.iloc[0]['1y']}, ignore_index=True)            

    print("----------- Check if all holdings are monitored --------")
    pprint.pprint(holdingsList)
    for a in holdingsList:
        if not holdingsList[a]:
            print(f"Check fund {a}")


    print("")
    print("====== Observations summary ============")
    print(pdSummary.sort_values(by=['Holding'], ascending=False).to_string())



        



