import pandas as pd
import glob, os
import pprint
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import datetime
import sys


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
        allFundsInf = allFundsInf.append(fundInf, ignore_index=True)

    print("-------- ALL ----------------")
    print(allFundsInf.shape)
    print(allFundsInf.columns)
    
    # convert date column to a datetime object
    allFundsInf['date'] = pd.to_datetime(allFundsInf['date']).dt.strftime("%m/%d/%y")  
    print(f"# Earliest date {allFundsInf['date'].min()}")
    


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
    groupedFundsList = allFundsInf.groupby('fundName', as_index=False)
    #print(groupedFundsList)
    for fund, frame in groupedFundsList:
        Holding = (fund in holdingsList)
        print(f"\n-=> Fund {fund} , holding {Holding} <=-")
        if Holding:
            holdingsList[fund] = True
        #print(frame.sort_values(by="date"))
        sortedFunds = frame.drop_duplicates(subset ="date", keep = False)
        #sortedFunds = frame1.sort_values(by="date", ascending=True)
        
        #sortedFunds.drop_duplicates(subset ="date", keep = False, inplace = True)

        # date  column becomes index column
        sortedFunds.set_index('date', drop=False, append=False, inplace=True, verify_integrity=False)
        sortedFunds.sort_index()
        # reverse order
        sortedFunds = sortedFunds.iloc[::-1]
        displayColumsList = ['Quartile', 'FERisk', '3m', '6m', '1y', '3y', '5y']
        
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
        print(f"[DBG] All Past Data {pastData}")
        
        #pastData = list(range(10))
        #print(pastData)
        
        P = 4
        weeksPeriod = 2*P-1
        #print(pastData[-weeksPeriod-1:-1])
        for ii in range(len(pastData)-weeksPeriod):
            #print(ii, -ii-weeksPeriod-1, -ii-1)
            
            predictivePast = pastData[-ii-weeksPeriod-1:-ii-1]
            print(predictivePast, pastData[-1-ii])   
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
                    desDataString += "Y[1]"
                    
                # create the data list for the prediction
                outDataList =  [*predictivePast]
                outDataList+= [*popt]
                outDataList+= [sortedFunds['3m'].iloc[ii+1]]
                outDataList+= [sortedFunds['6m_annual'].iloc[ii+1]]
                outDataList+= [Y]
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
                    evalFile.write(outDataString+"\n")
                    print(f"Eval Data: {outDataString}")
                else:
                    trainFile.write(outDataString+"\n")
                    print(f"Train Data: {outDataString}")
                
        
        #print(f"predictNext:{predictNext:.3f}>3m_annual:{sortedFunds['3m_annual'][0]:.3f}")
        #if predictNext>sortedFunds['3m_annual'][0] and \
        #    sortedFunds['3m_annual'][0]>sortedFunds['3m_annual'][1] and \
        #    sortedFunds['3m_annual'][1]>sortedFunds['3m_annual'][2]:
        #    print("Getting better !")


    #close files
    trainFile.close()
    evalFile.close()
    
    