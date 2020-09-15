import pandas as pd
import glob, os
import pprint
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import datetime
import sys

def dbgPrint(*args):
    ''' Debug function '''
    if True:
        print("[DBG]", *args)
    else:
        pass
 
def func(x, a, b, c, d):
    ''' ~x^3 curve     '''
    return a + b * x + c * x * x + d * pow(x,3)

def get_curve_fit(pastData):
    '''
        estimate coefficients of curve fit to ~x^3
    '''
 
    P=3    
    if len(pastData)<2*P-1:
        return False, None, 0

    # use just last 10 weeks
    if len(pastData)>10:
        pastData = pastData[-10:]

    xdata = np.linspace(P+1-len(pastData), P, len(pastData))
    #print(xdata)
    popt, pcov = curve_fit(func, xdata, pastData)
    
    # return: succesful, coefficients, predict next 
    return True, popt, func(P+1, *popt)


if __name__ == "__main__":
    
    print("# Set working directory.")
    os.chdir('C:\\Users\\tzurv\\python\\VScode\\scraper')

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

    print("# Load Funds Information =====================================================")
    # create empty dataframe
    allFundsInf = pd.DataFrame()
    for file in glob.glob("*_FundsInf.csv"):
        print(f"Loading {file}" )

        # loading examples
        #https://chrisalbon.com/python/data_wrangling/pandas_dataframe_importing_csv/
        fundInf = pd.read_csv(file, sep=',', dayfirst=True)

        # set datetime
        dtm = lambda x: datetime.strptime(x, "%d/%m/%y %H:%M")
        fundInf["date"] = fundInf["date"].apply(dtm)

        # get unique fund code
        fundCode = lambda P: P.split('/')[-2]
        fundInf["fundCode"] = fundInf["url"].apply(fundCode)
        #dbgPrint(fundInf.to_string())
        
        allFundsInf = allFundsInf.append(fundInf, ignore_index=True)

    print("-------- ALL ----------------")
    print(allFundsInf.shape)
    print(allFundsInf.columns)
    
    print("# Load Sector Information =====================================================")
    # create empty dataframe
    allSectorsInf = pd.DataFrame()
    for file in glob.glob("*_TrustNetSectors.csv"):
        print(f"Loading {file}" )

        sectorsInf = pd.read_csv(file, sep=',', dayfirst=True)
        dtm = lambda x: datetime.strptime(x, "%d/%m/%y %H:%M")
        sectorsInf["date"] = sectorsInf["date"].apply(dtm)
        
        # development code 
        #dbgPrint(sectorsInf.to_string())
        #dbgPrint(sectorsInf.loc[sectorsInf['sectorName'] == 'IA Flexible Investment'].index)
        #dbgPrint(sectorsInf.loc[ (sectorsInf['date'] == pd.to_datetime('2020-08-20 15:53:00', format='%Y-%m-%d %H:%M:%S', errors='ignore')) &\
        #                        (sectorsInf['sectorName'] == 'IA Flexible Investment') ])
        #dbgPrint(sectorsInf.loc[ (sectorsInf['date'].dt.date == pd.to_datetime('2020-08-20', format='%Y-%m-%d', errors='ignore')) &\
        #                        (sectorsInf['sectorName'] == 'IA Flexible Investment') ])
        #sys.exit(0)
        
        allSectorsInf = allSectorsInf.append(sectorsInf, ignore_index=True)

    print("-------- ALL ----------------")
    print(allSectorsInf.shape)
    print(allSectorsInf.columns)
    
    sectorSelectedColumns = ['1m', '3m', '6m', '1y', '3y', '5y']
    allSectorsMean = [x for x in allSectorsInf[sectorSelectedColumns].mean()]
    #allSectorsMean = allSectorsMean[1:]
    dbgPrint(f"# Sectors average {[ '%.2f' % elem for elem in allSectorsMean ] }")
        

    # convert date column to a datetime object
    #allFundsInf['date'] = pd.to_datetime(allFundsInf['date']).dt.strftime("%d/%m/%y") 
    print(f"Erliest Date {min(allFundsInf['date'])}")
    #############################################
    if False:
        earliest = None
        setDates = set()
        for d1 in allFundsInf['date']:
            if earliest is None or earliest>d1:
                print(earliest, "<-", d1)
                earliest = d1
            setDates.add(d1)
        print(setDates)
        print("selected: ", earliest)
        print(type(d1), type(allFundsInf['date'].iloc[0]))

        # print("-"*20)
        # earliest = None
        # for d1 in setDates:
            # print(d1)
            # dt1 = datetime.strptime(d1, "%d/%m/%y")
            # print(dt1.strftime('%B'))
            # if earliest is None or earliest>dt1:
                # print(earliest, "<-", dt1)
                # earliest = dt1
        # print("selected: ", earliest)
        sys.exit(0)
    #################################################
    


    #=============================
    #                      3 first rows and 5 columns
    #print(allFundsInf.iloc[:3, 0:5])
    #   Unnamed: 0            date                                       fundName  Quartile  FERisk
    #0           0  14/06/20 23:33  HSBC World Selection Cautious Portfolio C Acc       NaN    27.0
    #1           1  14/06/20 23:33                         Jupiter European I Acc       2.0    79.0
    #2           2  14/06/20 23:33                   Liontrust Global Alpha C Acc       1.0    89.0


    # ===========================
    # https://www.shanelynn.ie/summarising-aggregation-and-grouping-data-in-python-pandas/
    print(allFundsInf.groupby('Sector').Sector.count())
    print(allFundsInf.groupby(['fundName', 'date'])['date'].count())

    # Save data for model training
    now = datetime.now()
    current_time = now.strftime("%d/%m/%y %H:%M")
    print("Current Time =", current_time)
    dateStamp = now.strftime("%Y%m%d")
    TrainFileName = "C:\\Users\\tzurv\\python\\VScode\\scraper\\" + dateStamp + "_Train.csv"
    EvalFileName  = "C:\\Users\\tzurv\\python\\VScode\\scraper\\" + dateStamp + "_Eval.csv"
    descriptionFileName  = "C:\\Users\\tzurv\\python\\VScode\\scraper\\" + dateStamp + "_Des.csv"

    trainFile = open(TrainFileName, 'w')
    evalFile = open(EvalFileName, 'w')
    desFileFlag = True
    desFile = open(descriptionFileName, 'w')


    # group by fundname
    #groupedFundsList = allFundsInf.groupby('fundName', as_index=False)
    
    # group by fundname (code)
    groupedFundsList = allFundsInf.groupby('fundCode', as_index=False)
    
    #print(groupedFundsList)
    for fundCode, frame in groupedFundsList:
        fund = frame.fundName.iloc[0]
        Holding = (fund in holdingsList)
        print(f"\n-=> Fund {fund} , holding {Holding} <=-")
        if Holding:
            holdingsList[fund] = True

        # remove duplicate entries of dates for this fund            
        sortedFunds = frame.drop_duplicates(subset ="date", keep = False)
        
        # date  column becomes index column
        sortedFunds.set_index('date', drop=False, append=False, inplace=True, verify_integrity=False)
        sortedFunds.sort_index()

        # reverse order
        sortedFunds = sortedFunds.iloc[::-1]

        # get Fund sector
        fundSector = sortedFunds['Sector'][0]

        displayColumsList = ['Quartile', 'FERisk', '3m', '6m', '1y', '3y', '5y', 'Sector']
        
        # add more processed
        newColumnName = '3m_annual'
        displayColumsList.append(newColumnName)
        sortedFunds[newColumnName] = 100*(pow(1+sortedFunds['3m']/100.0,12/3) - 1.0)
        
        newColumnName = '6m_annual'
        displayColumsList.append(newColumnName)
        sortedFunds[newColumnName] = 100*(pow(1+sortedFunds['6m']/100.0,12/6) - 1.0)

        #newColumnName = '3y_annual'
        #displayColumsList.append(newColumnName)
        #sortedFunds[newColumnName] = 100*(pow(1+sortedFunds['3y']/100.0,1/3) - 1.0)

        #newColumnName = '5y_annual'
        #displayColumsList.append(newColumnName)
        #sortedFunds[newColumnName] = 100*(pow(1+sortedFunds['5y']/100.0,1/5) - 1.0)
        
        print(sortedFunds[displayColumsList].to_string())

        #print(sortedFunds['3m_annual'].to_string(index=False))
        pastData = sortedFunds['3m_annual'].tolist()
        pastData.reverse()
        dbgPrint(f"All Past Data {pastData}")
                
        # collect data dates
        dataDatesList =  [x.strftime('%Y-%m-%d') for x in sortedFunds.index]
        # development code
        if False:
            dbgPrint(f"Date: {dataDatesList[0]}, Sector: {sortedFunds['Sector'][0]}")
            dbgPrint(allSectorsInf['sectorName'])
            dbgPrint(allSectorsInf['sectorName'] == sortedFunds['Sector'][0])
            dbgPrint(allSectorsInf.loc[ (allSectorsInf['date'].dt.date == \
                                       pd.to_datetime(dataDatesList[0], format='%Y-%m-%d', errors='ignore')) & \
                                           (allSectorsInf['sectorName'] == fundSector ) ].to_string())
            
            #continue
            sys.exit(0)
        
        
        # History window size 
        P = 4
        weeksPeriod = 2*P-1
        
        # Moving history window
        for ii in range(len(pastData)-weeksPeriod):
            
            predictivePast = pastData[-ii-weeksPeriod-1:-ii-1]
            dbgPrint(predictivePast, pastData[-1-ii])   

            # get sector performance history vector
            sectorVec = allSectorsMean
            fundSectorPerformanceOnDate = allSectorsInf.loc[ (allSectorsInf['date'].dt.date == \
                                       pd.to_datetime(dataDatesList[ii], format='%Y-%m-%d', errors='ignore')) & \
                                           (allSectorsInf['sectorName'] == fundSector ) ]
            if len(fundSectorPerformanceOnDate):
                dbgPrint(fundSectorPerformanceOnDate[sectorSelectedColumns])
                sectorVec = fundSectorPerformanceOnDate.iloc[0][sectorSelectedColumns].tolist()
            
            status, popt, predictNext = get_curve_fit(pastData[-ii-weeksPeriod-1:-ii-1])
            if status:
                if False:
                    Y = 1
                    if pastData[-1-ii]<=0:
                        print(f"Negative: {pastData[-1-ii]}")
                        Y = 0
                    elif pastData[-2-ii]*0.9>pastData[-1-ii]:
                        print(f"Worst in 10% or more: {pastData[-2-ii]} -> {pastData[-1-ii]} ")
                        Y = 0
                else:
                    Y = pastData[-1-ii]
                
                ## add description for every data/feature entry 
                if desFileFlag:
                    desDataString  = f"predictivePast[{weeksPeriod}], "
                    desDataString += "fit[4], "
                    desDataString += "3m, "
                    desDataString += "6m_annual, "
                    desDataString += "Y[1], "
                    desDataString += "Sector[6]"
                    
                    
                # create the data list for the prediction
                outDataList =  [*predictivePast]
                outDataList+= [*popt]
                outDataList+= [sortedFunds['3m'].iloc[ii+1]]
                outDataList+= [sortedFunds['6m_annual'].iloc[ii+1]]
                outDataList+= [Y]
                outDataList+= sectorVec
                outDataString = str(outDataList).strip('[]')
                #outDataString = str([*predictivePast, *popt, sortedFunds['3m'].iloc[-ii-1], Y ]).strip('[]')

                if desFileFlag:
                    # Write to description/headr file and close
                    #desFile.write(f"predictivePast[{weeksPeriod}], fit[4], Y[1]")
                    desFile.write(f"{desDataString}\n")
                    desFile.close()
                    desFileFlag = False
                
                # write to a file
                if ii == 0:
                    evalFile.write(f"{fund}, {outDataString}\n")
                    print(f"#'{fund}' Eval Data: {outDataString}")
                else:
                    trainFile.write(outDataString+"\n")
                    dbgPrint(f"Train Data: {outDataString}")


    #close files
    trainFile.close()
    evalFile.close()
    
    