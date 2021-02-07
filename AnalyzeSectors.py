import pandas as pd
import glob
from datetime import datetime
import json

import warnings
warnings.filterwarnings('ignore')


def addPerformanceOrder(sectorsInf, columnName):
    #https://stackoverflow.com/questions/33165734/update-index-after-sorting-data-frame
    sortedByColumn =  sectorsInf.sort_values(columnName, ascending=False, ignore_index=True)
    sortedByColumn['O_'+columnName] = [i for i in range(sortedByColumn.shape[0])]
    
    #print(sortedByColumn.columns)
    #print(sortedByColumn[["O_1m","1m"]])
    #for sector in sortedByColumn.itertuples():
    #    print(sector)
    return sortedByColumn

def loadSectorInf():
    print("# Load Sector Information =====================================================")
    # create empty dataframe
    sectorSelectedColumns = ['1m', '3m', '6m', '1y', '3y', '5y']
    allSectorsInf = pd.DataFrame()
    totalSectorsInfFiles = 0
    file = None
    sectorTop5counter = {}
    allFilesList = glob.glob("*_TrustNetSectors.csv")
    for file in allFilesList[-12:]:
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
'''

if __name__ == "__main__":

    allSectorsInf, sectorTop5counter = loadSectorInf()
    print(json.dumps(sectorTop5counter, indent=2))
    