import pandas as pd
import glob, os
import pprint
from datetime import datetime
import sys

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":

    print("#=====================================================")
    print("-------- Loading fund information ('*_FundsInf.csv') ----------------")

    # Create empty dataframe to store all fund information
    allFundsInf = pd.DataFrame()
    loadedFile = 0
    last_file = None
    last_fundInf = None

    print(f"Current working directory: {os.getcwd()}")

    # Iterate over all '_FundsInf.csv' files in the current directory
    for file in glob.glob("*_FundsInf.csv"):
        last_file = file
        loadedFile += 1

        # Load fund information from the CSV file
        fundInf = pd.read_csv(file, sep=',')

        # Convert 'date' column to datetime format
        fundInf["date"] = fundInf["date"].apply(lambda x: datetime.strptime(x, "%d/%m/%y %H:%M"))

        # Extract fund code from 'url' column
        fundInf["fundCode"] = fundInf["url"].apply(lambda P: str(P.split('/')[-2]).lower())
        last_fundInf = fundInf

        # Append fund information to the main dataframe
        allFundsInf = allFundsInf.append(fundInf, ignore_index=True)

    print(f"# Total '*_FundsInf.csv' files loaded: {loadedFile}")
    print("\n")

    print(f"All Funds Information Columns: {allFundsInf.columns}")
    print("\n")

    # Gather holding information from the last loaded file
    print(f"# Gathering holding information from {last_file}....")

    holdingsList = {}  # Dictionary to track fund holdings (True/False)
    activeMonitoredList = []  # List of actively monitored fund codes

    # Iterate through the last loaded fund information
    for fund in last_fundInf.itertuples():
        activeMonitoredList.append(fund.fundCode)
        if fund.Hold:
            holdingsList[fund.fundName] = False  # Initialize holdings to False

    print("-------- Loading completed ----------------")
    print("\n")

    # Load sector information ('*_TrustNetSectors.csv')
    print("# Loading Sector Information")

    sectorSelectedColumns = ['1m', '3m', '6m', '1y', '3y', '5y']
    allSectorsInf = pd.DataFrame()  # Dataframe to store all sector information
    totalSectorsInfFiles = 0
    last_sectorsInf = None

    # Iterate over all '*_TrustNetSectors.csv' files
    for file in glob.glob("*_TrustNetSectors.csv"):
        totalSectorsInfFiles += 1

        # Load sector information from CSV file
        sectorsInf = pd.read_csv(file, sep=',', dayfirst=True)
        sectorsInf["date"] = sectorsInf["date"].apply(lambda x: datetime.strptime(x, "%d/%m/%y %H:%M"))

        # Remove rows with missing numerical data
        sectorsInf = sectorsInf.drop(sectorsInf[sectorsInf['1m'] == '-'].index)

        # Convert selected columns to float
        for col in sectorSelectedColumns:
            sectorsInf[col] = sectorsInf[col].astype(float)

        last_sectorsInf = sectorsInf
        allSectorsInf = allSectorsInf.append(sectorsInf, ignore_index=True)

    print("-------- ALL ----------------")
    print(f"# Total '*_TrustNetSectors.csv' files loaded: {totalSectorsInfFiles}")
    print(f"Latest file is: {last_file}")
    print(f"Count of funds per sector over the whole data collection period: {allSectorsInf.shape}")
    
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
        
    # collect sectors summary
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
    
    # Title
    print(f"{'sector':<60}: {'funds':>4} {'Cases':>6} {sectorSelectedColumns}")
    for sector in sectors_summary:
        indx = last_sectorsInf.index[last_sectorsInf['sectorName']==sector]
        if len(indx)>0:
            print(f"{sector:<60}: {sectors_summary[sector]['count']:>4} {sectors_summary[sector]['Weekly']:>6} [{last_sectorsInf.loc[indx, sectorSelectedColumns].to_string(index=False, header=False, col_space=7)}]")
        elif isinstance(sectors_summary[sector]['count'], int):
            print(f"{sector:<60}: {sectors_summary[sector]['count']:>4} {sectors_summary[sector]['Weekly']:>6}")
    print(f"-- End of sectors summary.")
    print("\n")
    

    # Group by sector and sort by 3m performance (descending)
    grouped_by_sector = allFundsInf.groupby('Sector')

    # Find the most recent date in the entire dataset
    most_recent_date = allFundsInf['date'].max()

    # Iterate over each sector
    for sector, sector_data in grouped_by_sector:
        
        # Filter data for the most recent date
        latest_date_in_sector = sector_data['date'].max()

        # Check if the sector was updated on the most recent date
        if not latest_date_in_sector == most_recent_date:
            continue
           
        latest_sector_data = sector_data[sector_data['date'] == latest_date_in_sector]

        # Remove duplicates from the latest sector data (assuming all columns matter for uniqueness)
        latest_sector_data = latest_sector_data.drop_duplicates() 

        # Sort by 3m performance in descending order and get top 3
        top_3_funds = latest_sector_data.sort_values('3m', ascending=False).head(3)
        
        # Filter held funds: Hold is '1' and Holding% is not nan
        held_funds = latest_sector_data[(latest_sector_data['Hold'] == 1) & (~latest_sector_data['Holding%'].isna())]

        # Combine top 3 and held funds, removing duplicates
        combined_funds = pd.concat([top_3_funds, held_funds]).drop_duplicates()
        
        # Check if any fund in top_3_funds has 'Hold' value of 1
        has_holding = combined_funds['Hold'].isin([1]).any()

        # Print sector and date information
        print(f"Sector: {sector}\t Sector-Holding:{has_holding}")
        print(f"Date: {latest_date_in_sector.date()}")

        # Print column headers
        print("Name                                  Quartile  FERisk    3m    6m    1y    3y    5y  Hold         price             Holding%")

        # Print information for each of the top 3 funds
        for _, fund_data in combined_funds.iterrows():
            print(f"{fund_data['fundName']:<60} {fund_data['Quartile']:<9} {fund_data['FERisk']:<7} {fund_data['3m']:<6} {fund_data['6m']:<6} {fund_data['1y']:<6} {fund_data['3y']:<6} {fund_data['5y']:<6} {fund_data['Hold']:<6} {fund_data['price']:<18} {fund_data['Holding%']}")

        print("\n")  # Add spacing between sectors


    #https://realpython.com/pandas-groupby/
    #groupedFundsList = allFundsInf.groupby(['Sector','fundName', 'date'], as_index=False)
    #print(groupedFundsList)
    #for (sector, fund, date, frame) in groupedFundsList:
    #    print(sector, fund, date)
    
    # create report
    COLUMN_NAMES=['fundName', 'Holding', 'from_date', '#weeks', 'past_held', 'cur_price', 'purchase_price', '3m', '6m', '1y', 'worsenQuartile', 'worsenFERisk', 'worse3mThanSector']
    pdSummary = pd.DataFrame(columns=COLUMN_NAMES)
    
    # group by fundname (code)
    groupedFundsList = allFundsInf.groupby('fundCode', as_index=False)
    
    # print(groupedFundsList)
    for fundCode, frame in groupedFundsList:

        # skip funds not on the current excel list
        if not fundCode in activeMonitoredList:
            continue

        sortedFunds = frame.drop_duplicates(subset ="date", keep = 'first')

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

        Holding = (fund in holdingsList)
        print(f"\n-=> Fund {fund} , holding {Holding} <=-")
        print(f"\t\tCode:{fundCode}\tSector: {fundSector}")
        if Holding:
            holdingsList[fund] = True

        worse3MthanSector = False
        fundSectorPerformanceOnDate = allSectorsInf.loc[ (allSectorsInf['date'].dt.date == \
                                   pd.to_datetime(latestDate, format='%Y-%m-%d', errors='ignore')) & \
                                       (allSectorsInf['sectorName'] == fundSector ) ]

        try:
            if len(fundSectorPerformanceOnDate):
                sector3m = float(fundSectorPerformanceOnDate[sectorSelectedColumns]['3m'])
                fund3m = float(sortedFunds.iloc[0]['3m'])
                worse3MthanSector = bool(fund3m<sector3m)
        except Exception as ex:
            pass
        
        
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
        hold_inf = sortedFunds['Hold'].value_counts()        
        weeks_held = 0
        from_date = None
        past_held = False
        purchase_price = None
        if True in hold_inf.index:
            for i in range(0, len(sortedFunds.index)):
                if sortedFunds.Hold.iloc[i]:
                    from_date = sortedFunds.index[i].date()
                    weeks_held += 1
                    purchase_price = sortedFunds.price.iloc[i]
                else:
                    break
                    
            past_held = True if True in hold_inf.index else False

        if from_date is not None:
            print(f"# Held since {from_date} for ~{weeks_held} weeks, ~purchase price {purchase_price}")
        print(sortedFunds[:20][['Quartile', 'FERisk', '3m', '6m', '1y', '3y', '5y', 'Hold', 'price' , 'Holding%']].to_string())
        
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
                                      'cur_price': sortedFunds.iloc[0]['price'],
                                      'purchase_price': purchase_price,
                                      'past_held': past_held,
                                      '3m':sortedFunds.iloc[0]['3m'],
                                      '6m':sortedFunds.iloc[0]['6m'], 
                                      '1y':sortedFunds.iloc[0]['1y'],
                                      'worsenQuartile':worsenQuartile, 
                                      'worsenFERisk':worsenFERisk,
                                      'worse3mThanSector':worse3MthanSector
                                      }, ignore_index=True)            

    print("#--------------------------------------------------------#")
    print("#----------- Check if all holdings are monitored --------#")
    pprint.pprint(holdingsList)
    for a in holdingsList:
        if not holdingsList[a]:
            print(f"Check fund {a}")

    print("\n")
    print("====== Observations summary ============")
    print(pdSummary.sort_values(by=['Holding'], ascending=False).to_string())


