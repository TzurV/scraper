# my source is https://www.pluralsight.com/guides/guide-scraping-dynamic-web-pages-python-selenium


#import chromedriver-binary
#path = cdi.install(file_directory='c:\\data\\chromedriver\\', verbose=True, chmod=True, overwrite=False, version=None)
#print('Installed chromedriver to path: %s' % path)

# coppied from
# https://pypi.org/project/chromedriver-binary/
# https://www.youtube.com/watch?v=FFDDN1C1MEQ - Learn Selenium Python
# https://selenium-python.readthedocs.io/api.html#module-selenium.webdriver.remote.webdriver
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

#import chromedriver_binary # Adds chromedriver binary to path

import pandas as pd
from datetime import datetime
import re


import sys
import time

#============================================
# (Scraper) C:\Users\...\scraper>pip freeze  > requirements.txt

#https://pip.pypa.io/en/stable/reference/pip_freeze/
#$ env1/bin/pip freeze > requirements.txt
#$ env2/bin/pip install -r requirements.txt


# ----------------------------- example ------------------------
# driver = webdriver.Chrome()
# driver.get("http://www.python.org")
# assert "Python" in driver.title


# # 20-may-2020

# elem = driver.find_element_by_name("q")
# elem.clear()
# elem.send_keys("start")
# driver.find_element_by_name("submit").click()

#================================= End Example =================================

#------------------------------------------------------------------------
'''

Getting Started With openpyxl

https://realpython.com/openpyxl-excel-spreadsheets-python/
pip install openpyxl
'''


#from openpyxl import Workbook
from openpyxl import load_workbook

class MyHoldingsExcell:
    def __init__(self):
        
        self.AllHoldingsFile = r"E:\Family\צור\bank and money\Investments_reports\AllHoldings_Updated.xlsx"
        print(f"# getting iformation from {self.AllHoldingsFile}.")
        self.workbook = load_workbook(filename=self.AllHoldingsFile, data_only=True)
        pass

    def getTrackingListURLs(self):
        TrackingList_sheet = self.workbook["TrackingList"]
        trackingURLsColumn = TrackingList_sheet["F:F"]        
        holdingColumn = TrackingList_sheet["G:G"]
        
        trackingURLsList = []
        totHoldings = 0
        for indx in range(3, len(trackingURLsColumn)):
            haveIt = False
            if holdingColumn[indx].value :
                haveIt = True
                totHoldings += 1
                
            trackingURLsList.append({"URL":trackingURLsColumn[indx].value,
                                     "Hold":haveIt})
           #print(f"{haveIt} {trackingURLsColumn[indx].value}")
        
        print(f"# URLs summary: {len(trackingURLsList)} loaded and listed holding {totHoldings}")
        print(trackingURLsList[-5:-1])
        return trackingURLsList
    

#------------------------------------------------------------------------
'''
'''
# https://www.techbeamers.com/handling-html-tables-selenium-webdriver/
# https://chercher.tech/python/table-selenium-python

# globals
column_names = ["date", "fundName", "Quartile", "FERisk", "3m", "6m", "1y", "3y", "5y", "url", "Hold"]
Empty_fund_df = pd.DataFrame(columns = column_names)


#--------------------------------
# from selenium import webdriver
class WebTable:
    def __init__(self, webtable):
       self.table = webtable

    def get_row_count(self):
      return len(self.table.find_elements_by_tag_name("tr")) - 1

    def get_column_count(self):
        return len(self.table.find_elements_by_xpath("//tr[2]/td"))

    def get_table_size(self):
        return {"rows": self.get_row_count(),
                "columns": self.get_column_count()}

    def row_data(self, row_number):
        if(row_number == 0):
            raise Exception("Row number starts from 1")

        row_number = row_number
        row = self.table.find_elements_by_xpath("//tr["+str(row_number)+"]/td")
        rData = []
        for webElement in row :
            rData.append(webElement.text)

        return rData

    def column_data(self, column_number):
        col = self.table.find_elements_by_xpath("//tr/td["+str(column_number)+"]")
        rData = []
        for webElement in col :
            rData.append(webElement.text)
        return rData

    def get_all_data(self):
        # get number of rows
        noOfRows = len(self.table.find_elements_by_xpath("//tr")) -1
        # get number of columns
        noOfColumns = len(self.table.find_elements_by_xpath("//tr[2]/td"))
        allData = []
        # iterate over the rows, to ignore the headers we have started the i with '1'
        for i in range(2, noOfRows):
            # reset the row data every time
            ro = []
            # iterate over columns
            for j in range(1, noOfColumns) :
                # get text from the i th row and j th column
                ro.append(self.table.find_element_by_xpath("//tr["+str(i)+"]/td["+str(j)+"]").text)

            # add the row data to allData of the self.table
            allData.append(ro)

        return allData

    def presence_of_data(self, data):

        # verify the data by getting the size of the element matches based on the text/data passed
        dataSize = len(self.table.find_elements_by_xpath("//td[normalize-space(text())='"+data+"']"))
        presence = False
        if(dataSize > 0):
            presence = True
        return presence

    def check_cell_data(self, row_number, column_number):
        """ check if cell exist """
        #https://stackoverflow.com/questions/38022658/selenium-python-handling-no-such-element-exception/38023345
        if(row_number == 0):
            raise Exception("Row number starts from 1")

        row_number = row_number + 1
        try:
            self.table.find_element_by_xpath("//tr["+str(row_number)+"]/td["+str(column_number)+"]").text
    
        except NoSuchElementException:  #spelling error making this code not work as expected
            return False

        return True

    def get_text(self):
        return self.table.text

    def get_cell_data(self, row_number, column_number):
        if(row_number == 0):
            raise Exception("Row number starts from 1")

        row_number = row_number + 1
        cellData = self.table.find_element_by_xpath("//tr["+str(row_number)+"]/td["+str(column_number)+"]").text
        return cellData

def is_number(n):
    try:
        float(n)   # Type-casting the string to `float`.
                   # If string is not a valid `float`, 
                   # it'll raise `ValueError` exception
    except ValueError:
        return False
    return True

class trustnetInf:
    """ get funds information from trustnet website     """

    def __init__(self):
        self._first = True
        self.driver = webdriver.Chrome()
        self.driver.implicitly_wait(30)


    def getFundInf_v2(self, fundUrl, openAndReturn=False):

        if not self._first:
            # open new blank tab
            #print(len(self.driver.window_handles))
            self.driver.switch_to.window(self.driver.window_handles[0])
            self.driver.execute_script("window.open();")
            time.sleep(5)

            # switch to the new window which is second in window_handles array
            self.driver.switch_to.window(self.driver.window_handles[-1])   

        status = self.driver.get(fundUrl)
        self.driver.implicitly_wait(30)
        print("Get status: ", status)

        if self._first :
            #<button tabindex="0" type="button" mode="primary" class="sc-bwzfXH bbIVrv">ACCEPT ALL</button>
            #XPATH: "/html/body/div[1]/div/div/div/div[2]/div/button[2]"
            #https://selenium-python.readthedocs.io/getting-started.html Section:  2.2. Example Explained
            
            #  click 'Accept ALL'
            elem = self.driver.find_element_by_xpath(u'/html/body/div[1]/div/div/div/div[2]/div/button[2]')
            elem.click()

            #element = WebDriverWait(self.driver, 5).until(EC.element_to_be_clickable((By.XPATH, u"/html/body/div[1]/div/div/div/div[1]/div/div[2]/button[2]")))
            #element.click()
            #print("TZUR AABC")
            # Click 'SAVE AND EXIT'
            #elem = self.driver.find_element_by_xpath(u"/html/body/div[1]/div/div/div/div[3]/div[1]/button[2]")
            #elem.click()
            
            elem = self.driver.find_element_by_xpath(u"/html/body/div[1]/div/div/div/div[3]/div[2]/button")
            elem.click()

            #Select 'I am a private investor'
            elem = self.driver.find_element_by_xpath("/html/body/user-type-popup/div[1]/div[3]/div/div[1]/p[5]/label/span")
            elem.click()

            # Click 'I agree'
            elem = self.driver.find_element_by_xpath("/html/body/user-type-popup/div[1]/div[3]/div/div[2]/p[3]")
            elem.click()

            
            #input("\n ------------------ \n >> Set Agree options befor provessing: \n ")
            self._first = False
            

        _statusOK = True

        # open web page and return
        if openAndReturn:
            time.sleep(60)
            return _statusOK, self.driver
    
        print("Check point 1 ! ")

        # dictionary for gathering information from web page
        fundDict =  {   "date":"NA",
                        "fundName": "NA",
                        "3m": "NA",
                        "6m": "NA",
                        "1y": "NA",
                        "3y": "NA",
                        "5y": "NA",
                        "Quartile": "NA",
                        "FERisk": "NA",
                        "Sector": "NA",
                        "SectorUrl": "NA",
                        "Hold": False}

        print("Check point 2 ! ")
        time.sleep(5)
        try:
            _notFoundTable = True
            self.driver.implicitly_wait(1)
            _AllTableElement = self.driver.find_elements_by_class_name("data_table")
            
            for _TableElement in _AllTableElement:
                #print(type(_TableElement))

                #if re.search('Quartile Ranking', _TableElement.text):
                if re.search('3 m 6 m', _TableElement.text):
                    #print(_TableElement.text)
                    # found table
                    _notFoundTable = False
                    #print(">> found the table! ")

                    # get fund name
                    _fundName = self.driver.find_element_by_class_name("fundName")
                    fundDict["fundName"] = _fundName.text

                    break
                 
        except NoSuchElementException:
            print(f"webpage {fundUrl} don't include required performance table")
            _statusOK = False
            
        except Exception as ex:
            print(ex) 
            _statusOK = False

        # gather information
        _found_3m_6m = False
        if not _notFoundTable and _statusOK:
            l = 0

            for line in _TableElement.text.split('\n'):

                if  re.search('3 m 6 m', line):
                    _found_3m_6m = True
                    continue
                elif not _found_3m_6m:
                    continue

                l += 1
                valuesList = line.split(' ')
                
                #print(l, " ",valuesList)

                if l == 1:
                    for p, key in zip(range(5), ["3m", "6m", "1y", "3y", "5y"]):
                        #print(p, " ", key)
                        if is_number(valuesList[p]):
                            fundDict[key] = float(valuesList[p])                

                if re.search('Quartile Ranking', line):
                    if is_number(valuesList[2]):
                        fundDict["Quartile"] = int(valuesList[2])

            print("\t>>> Got fund Sector ! ")
            try:
                # look for sector
                # from: https://stackoverflow.com/questions/54862426/python-selenium-get-href-value
                
                elems = self.driver.find_elements_by_xpath("//span//a[contains(text(),'(View sector)')]")
                #print(elems[0].get_attribute("href"))
                fundDict["SectorUrl"] = elems[0].get_attribute("href")

                _sectorEle = self.driver.find_element_by_class_name("view-sector")
                #print("#"+_sectorEle.text+"#")
                _sector = re.findall('Sector: (.*) \\(View sector\\)', _sectorEle.text)
                #print(_sector)
                fundDict["Sector"] = _sector

            except Exception as ex:
                print("Sector: ", ex) 
                #_statusOK = False

            print("\t>>> Got performance ! ")
            try:
            #     #<span class="risk_score">72</span>
                _FERisk = self.driver.find_element_by_class_name("risk_score")
                fundDict["FERisk"] = int(_FERisk.text)
                print("\t\t>>>> Got Risk score ! ")
        
            except NoSuchElementException:  
                pass
            
            # success 
            finally:
                # close tab 
                # (source: https://medium.com/@pavel.tashev/python-and-selenium-open-focus-and-close-a-new-tab-4cc606b73388)
                self.driver.close()
                time.sleep(5)
                # return connected information in dataframe
                return True, pd.DataFrame(fundDict, index=[0])

        # failed 
        # close tab 
        # (source: https://medium.com/@pavel.tashev/python-and-selenium-open-focus-and-close-a-new-tab-4cc606b73388)
        self.driver.close()
        time.sleep(5)

        # Create empty dataFrame
        _fundInf = Empty_fund_df.copy()
        
        return False, _fundInf


'''
'''
if __name__ == "__main__":

    #----------------------
    # Get current time date  
    #https://docs.python.org/3/library/datetime.html
    now = datetime.today()

    current_time = now.strftime("%d/%m/%y %H:%M")
    print("Current Time =", current_time)
    dateStamp = now.strftime("%Y%m%d")

    # collect all funds information
    allFundsInf = Empty_fund_df.copy()


    # get url list from trcking urls from excel
    myHoldingsExcell = MyHoldingsExcell()
    trackingURLsList = myHoldingsExcell.getTrackingListURLs()    

    #============
    # start chrom
    
    ChromeInstance = trustnetInf()

    if True:
        # SECTOR ANALYSIS
        print("Loading sectors performance")
        url = "https://www.trustnet.com/fund/sectors/performance?universe=O"
        _statusOK, chrom_driver = ChromeInstance.getFundInf_v2(url, openAndReturn=True)
        if _statusOK:
            print("Get Sectors data.")

            try:
                
                _notFoundTable = True
                chrom_driver.implicitly_wait(1)
                _AllTableElement = chrom_driver.find_elements_by_class_name("data_table")
                
                column_names = ["date", "sectorName", "1m", "3m", "6m", "1y", "3y", "5y"]
                sectors_df = pd.DataFrame(columns = column_names)

                for _TableElement in _AllTableElement:
                    #print(type(_TableElement), _TableElement.text)

                    if re.search('Rank Sector Name', _TableElement.text):
                        #print(_TableElement.text)
                        webTable = WebTable(_TableElement)
                        print(webTable.get_table_size())
                        for r in range(webTable.get_row_count()):
                            rowDataList = webTable.row_data(r+1)[:8]
                            # switch index by date
                            rowDataList[0] = current_time

                            # append data
                            df_length = len(sectors_df)
                            sectors_df.loc[df_length] = rowDataList

                        print(sectors_df.shape)
                        print(sectors_df)

                        # save all funds  information 
                        ## https://chrisalbon.com/python/data_wrangling/pandas_dataframe_importing_csv/
                        fileName = "C:\\Users\\tzurv\\python\\VScode\\scraper\\" + dateStamp + "_TrustNetSectors.csv"
                        print(f"Saving Sectors information to {fileName}")
                        sectors_df.to_csv(fileName, sep=',', float_format='%.2f')

                        # done gathering information, exit loop 
                        break
                    
            except NoSuchElementException:
                print(f"webpage {url} don't include required performance table")
                _statusOK = False
                
            except Exception as ex:
                print(ex) 
                _statusOK = False
       
    #sys.exit(0)
    
    # for backwards compatibility, create list from the file
    if False:
        # clear list 
        trackingURLsList.clear()
        
        # don't include information on holdings
        with open('FundsUrls.txt', 'r') as fh:
            for url in fh:
                trackingURLsList.append({"URL":url,
                                         "Hold":False})

        
    # loop over a list in a file
    totURLs = 0
    totSuccessful = 0
    failedURLs = list()
    for URLinf in trackingURLsList:
        url = URLinf['URL']
        
        totURLs += 1 
        url = url.rstrip("\n")
        print(f"# fetching: {url}") 
        
        reTries = 1
        maxNtime = 3
        while reTries<maxNtime:
            #print(f"# fetching: {url}")
            Status, fundInf = ChromeInstance.getFundInf_v2(url)
            reTries += 1
            
            if Status and not fundInf.empty:
                totSuccessful += 1
                fundInf.loc[0, 'date'] = current_time
                fundInf.loc[0, 'url'] = url
                fundInf.loc[0, 'Hold'] = URLinf['Hold']
                
                allFundsInf = allFundsInf.append(fundInf, ignore_index=True)
                print(allFundsInf)
                reTries=100
                
                if url in failedURLs:
                    failedURLs.remove(url)
                
            elif reTries==maxNtime:
                if not url in failedURLs:
                    failedURLs.append(url)
                    trackingURLsList.append(URLinf)
                    
            else:
                print(f"# Trying again {reTries} to fetch data ")
                
    # save all funds  information 
    ## https://chrisalbon.com/python/data_wrangling/pandas_dataframe_importing_csv/
    fileName = "C:\\Users\\tzurv\\python\\VScode\\scraper\\" + dateStamp + "_FundsInf.csv"
    try:
        print(f"Saving information to {fileName}")    
        allFundsInf.to_csv(fileName, sep=',', float_format='%.2f')
    except:
        print(f"Tryin AGain to Save information to {fileName}")    
        allFundsInf.to_csv(fileName+".II", sep=',', float_format='%.2f')

    if not totURLs == totSuccessful:
        print('#'*50)
        print(f"# Not all data collected. File includes {totSuccessful} out of {totURLs} urls in the list #")
        print('#'*50)
        print(failedURLs)






