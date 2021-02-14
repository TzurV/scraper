import matplotlib.pyplot as plt
import pandas as pd
import glob
from datetime import datetime
import json
import sys
# https://matplotlib.org/stable/gallery/misc/multipage_pdf.html
from matplotlib.backends.backend_pdf import PdfPages


# https://matplotlib.org/2.0.2/examples/lines_bars_and_markers/marker_reference.html
from six import iteritems
from matplotlib.lines import Line2D

import warnings
warnings.filterwarnings('ignore')

now = datetime.now()

current_time = now.strftime("%d/%m/%y %H:%M")
print("Current Time =", current_time)
dateStamp = now.strftime("%Y%m%d")


def addPerformanceOrder(sectorsInf, columnName):
    #https://stackoverflow.com/questions/33165734/update-index-after-sorting-data-frame
    sortedByColumn =  sectorsInf.sort_values(columnName, ascending=False, ignore_index=True)
    sortedByColumn['O_'+columnName] = [i for i in range(sortedByColumn.shape[0])]
    
    #print(sortedByColumn.columns)
    #print(sortedByColumn[["O_1m","1m"]])
    #for sector in sortedByColumn.itertuples():
    #    print(sector)
    return sortedByColumn

def loadSectorInf(historyWeeks = 8):
    print("# Load Sector Information")
    # create empty dataframe
    sectorSelectedColumns = ['1m', '3m', '6m', '1y', '3y', '5y']
    allSectorsInf = pd.DataFrame()
    totalSectorsInfFiles = 0
    file = None
    sectorTop5counter = {}
    allFilesList = glob.glob("*_TrustNetSectors.csv")
    for file in allFilesList[-historyWeeks:]:
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
        
        sectorsInf = addPerformanceOrder(sectorsInf, '1m')
        #print(sectorsInf[["O_1m","1m"]])
        
        # count number of times sector was in the best 5
        for sector in sectorsInf.loc[0:4]['sectorName']:
            if sector in sectorTop5counter:
                sectorTop5counter[sector] += 1
            else:
                sectorTop5counter[sector] = 1
        
        # add date information to global dataframe
        allSectorsInf = allSectorsInf.append(sectorsInf, ignore_index=True)

    print("-------- ALL ----------------")
    print(f"# Total '*_TrustNetSectors.csv' files loaded {totalSectorsInfFiles}")
    print(f"Latest file is {file}")
    #print(allSectorsInf.shape)
    #print(allSectorsInf.columns)
    #sys.exit(0)
    
    # sort by count
    sectorTop5counter = {k: v for k, v in sorted(sectorTop5counter.items(), key=lambda x: x[1], reverse=True)}
    return allSectorsInf, sectorTop5counter
    
'''
get information on what sectors are my holdings
'''
def loadHoldingsSectors():
    fundsInfFiles = glob.glob("*_FundsInf.csv")
    lastOne = fundsInfFiles[-1:][0]
    print(f"Loading {lastOne} for creating the sector holding list.")

    fundInf = pd.read_csv(lastOne, sep=',')

    dtm = lambda x: datetime.strptime(x, "%d/%m/%y %H:%M")
    fundInf["date"] = fundInf["date"].apply(dtm)

    #  count how many holding funds are in each sector
    holdingSectorsList = {}
    for fund in fundInf.itertuples():
        if fund.Hold:
            if not fund.Sector in holdingSectorsList:
                holdingSectorsList[fund.Sector] = 1
            else:
                holdingSectorsList[fund.Sector] += 1

    print(json.dumps(holdingSectorsList, indent=2))
    return holdingSectorsList


def legendString(sectorName, suffix, holdingSectorsList):
    colName = sectorName + suffix
    colName = colName.replace(' ','_')
    if sectorName in holdingSectorsList:
        colName += f"({holdingSectorsList[sectorName]})"
    return colName

def getPlotInformation(allSectorsInf, sectorTop5counter, holdingSectorsList):
    ''' 
    collect passed information on top 5 
    '''

    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/marker_reference.html
    # Filter out filled markers and marker settings that do nothing.
    # We use iteritems from six to make sure that we get an iterator
    # in both python 2 and 3
    # unfilled_markers = [m for m, func in iteritems(Line2D.markers)
    #                     if func != 'nothing' and m not in Line2D.filled_markers]
    
    filled_markers = [m for m, func in iteritems(Line2D.markers)
                        if func != 'nothing' and m in Line2D.filled_markers]
    
    markers = filled_markers
    
    line_types = ['-','--','-.',':']
    

    # group by fundname (code)
    groupedFundsList = allSectorsInf.groupby('sectorName', as_index=False)

    # create pdf with multiple pages
    pdfFileName = dateStamp+'_SectorsPerformance_all.pdf'
    with PdfPages(pdfFileName) as pdf:
        
        
        # create a unique list of sectors in the top 5 and holdings
        sectorsToReport = set(list(holdingSectorsList)+list(sectorTop5counter))
        
        # based on the example in https://matplotlib.org/stable/tutorials/text/text_intro.html
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # fig.subplots_adjust(top=0.85)
        ax.set_title('Sector Name (number of holdings)       <in top 5 list>')

        # Set both x- and y-axis limits to [0, 10] instead of default [0, 1]
        ax.axis([0, 10, 0, 10])  
        
        dY = 9.0 / float(len(sectorsToReport))
        Y = 9
        for indx, sector in enumerate(sectorsToReport):
                
            markerIndx = indx % len(markers)
            lineStyle  = indx % len(line_types)
    
            holdingsCount = 0
            if sector in holdingSectorsList:
                holdingsCount = holdingSectorsList[sector]
                
            if sector in sectorTop5counter and holdingsCount>0:
                ax.text(0.2, Y, f"{sector}:{holdingsCount}", 
                        fontsize=10, fontweight='bold')
            else:
                ax.text(0.2, Y, f"{sector}:{holdingsCount}", 
                        fontsize=10)
    
            Holding = '-'
            if sector in sectorTop5counter:
                Holding = f'{sectorTop5counter[sector]} weeks'
            ax.text(8, Y, f"{Holding}", 
                    fontsize=10, fontweight='bold')
                
            ax.plot([7.2, 7.8],[Y,Y],                     
                    markers[markerIndx]+line_types[lineStyle])
                    # style=line_types[lineStyle] )
            Y -= dY

        pdf.savefig()  # saves the current figure into a pdf page
        plt.show()
        plt.close()

        # leave just the ones that are at least half than the best count
        # the list is sorted
        if True:
            for indx, name in enumerate(list(sectorTop5counter)):
                if indx==0:
                    bestCount = int(sectorTop5counter[name]/2)
                    continue
                if sectorTop5counter[name]<bestCount:
                    del sectorTop5counter[name]

        # ------------------------
        ax  = plt.gca()
        markerIndx = 0
        lineStyle = 0
        for sectorName, frame in groupedFundsList:
        
            if not sectorName in sectorTop5counter:
                continue
            
            # set marker and line style
            markerIndx += 1
            markerIndx %= len(markers)
            
            lineStyle += 1
            lineStyle %= len(line_types)
    
            print(f"# processing {sectorName}")
            sortedFunds = frame.drop_duplicates(subset ="date", keep = False)
            
            # date  column becomes index column
            sortedFunds.set_index('date', drop=False, append=False, inplace=True, verify_integrity=False)
            sortedFunds.sort_index()
            
            # sector order
            colName = legendString(sectorName, "-O_1m", holdingSectorsList)           
            sortedFunds.rename(columns={"O_1m":colName}, inplace=True)
    
            sortedFunds.plot(kind='line', x='date',
                             marker=markers[markerIndx],
                             style=line_types[lineStyle],
                             y=colName, ax=ax)        
                
        # https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.legend.html
        # ax.legend(loc='best', fontsize='x-small', ncol=2)
        # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot/43439132#43439132
        ax.legend(bbox_to_anchor=(-0.1,0.89,1.2,0.3), loc="lower left",
                mode="expand", borderaxespad=0, ncol=2, fontsize='x-small')
        
        ax.set_ylabel("1m Order")
        plt.xticks(fontsize=8) 
        plt.grid(True)
        # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html
    
        pdf.savefig()  # saves the current figure into a pdf page
    
        # https://stackoverflow.com/questions/51203839/python-saving-empty-plot/51203929
        plt.show()
        plt.close()
    
        # ------------------------
        #  1month performance   
        ax  = plt.gca()
        markerIndx = 0
        lineStyle = 0
        for sectorName, frame in groupedFundsList:
        
            if not sectorName in sectorTop5counter:
                continue
            
            # set marker
            markerIndx += 1
            markerIndx %= len(markers)
            lineStyle += 1
            lineStyle %= len(line_types)
    
            print(f"# processing {sectorName}")
            sortedFunds = frame.drop_duplicates(subset ="date", keep = False)
            
            # date  column becomes index column
            sortedFunds.set_index('date', drop=False, append=False, inplace=True, verify_integrity=False)
            sortedFunds.sort_index()
    
            colName = legendString(sectorName, "-1m", holdingSectorsList) 
            sortedFunds.rename(columns={"1m":colName}, inplace=True)
            sortedFunds.plot(kind='line', x='date',
                             marker=markers[markerIndx],
                             style=line_types[lineStyle],
                             y=colName, ax=ax)        
    
        # ax.legend(loc='best', fontsize='x-small', ncol=2)
        ax.legend(bbox_to_anchor=(-0.1,0.89,1.2,0.3), loc="lower left",
                mode="expand", borderaxespad=0, ncol=2, fontsize='x-small')
        ax.set_ylabel("1m Performance")
        plt.xticks(fontsize=8) 
        plt.grid(True)

        pdf.savefig()  # saves the current figure into a pdf page
        plt.show()
        plt.close()

        # ------------------------
        #  3month performance   
        ax  = plt.gca()
        markerIndx = 0
        lineStyle = 0
        for sectorName, frame in groupedFundsList:
        
            if not sectorName in sectorTop5counter:
                continue
            
            # set marker
            markerIndx += 1
            markerIndx %= len(markers)
            lineStyle += 1
            lineStyle %= len(line_types)
    
            print(f"# processing {sectorName}")
            sortedFunds = frame.drop_duplicates(subset ="date", keep = False)
            
            # date  column becomes index column
            sortedFunds.set_index('date', drop=False, append=False, inplace=True, verify_integrity=False)
            sortedFunds.sort_index()
    
            colName = legendString(sectorName, "-3m", holdingSectorsList) 
            sortedFunds.rename(columns={"3m":colName}, inplace=True)
            sortedFunds.plot(kind='line', x='date',
                             marker=markers[markerIndx],
                             style=line_types[lineStyle],
                             y=colName, ax=ax)        
    
        # ax.legend(loc='best', fontsize='x-small', ncol=2)
        ax.legend(bbox_to_anchor=(-0.1,0.89,1.2,0.3), loc="lower left",
                mode="expand", borderaxespad=0, ncol=2, fontsize='x-small')
        ax.set_ylabel("3m Performance")
        plt.xticks(fontsize=8) 
        plt.grid(True)

        pdf.savefig()  # saves the current figure into a pdf page
        plt.show()
        plt.close()


        # ========================== 
        # Finalize PDF
        # We can also set the file's metadata via the PdfPages object:
        d = pdf.infodict()
        d['Title'] = 'Multipage PDF Example'
        d['Author'] = 'Tzur Vaich'
        d['CreationDate'] = datetime.now()
   

'''
'''

if __name__ == "__main__":

    holdingSectorsList = loadHoldingsSectors()
        
    allSectorsInf, sectorTop5counter = loadSectorInf(historyWeeks = 8)
    print(json.dumps(sectorTop5counter, indent=2))
    
    if not len(allSectorsInf):
        print("# Check working directory, no data was found. Aborting.")
        sys.exit(0)
        
        
    getPlotInformation(allSectorsInf, sectorTop5counter, holdingSectorsList)
    