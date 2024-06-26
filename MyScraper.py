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
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys 

# import Action chains 
from selenium.webdriver.common.action_chains import ActionChains

#import chromedriver_binary # Adds chromedriver binary to path

import pandas as pd
from datetime import datetime
import re

import sys
import time

from openpyxl import load_workbook

class MyHoldingsExcell:
    def __init__(self):
        
        self.AllHoldingsFile = r"D:\Family\צור\bank and money\Investments_reports\AllHoldings_Updated.xlsx"
        #self.AllHoldingsFile = r"C:\Family\Drive(D)\Family\צור\bank and money\Investments_reports\AllHoldings_Updated.xlsx"
        print(f"# getting iformation from {self.AllHoldingsFile}.")
        self.workbook = load_workbook(filename=self.AllHoldingsFile, data_only=True)
        pass

    def getTrackingListURLs(self):
        TrackingList_sheet = self.workbook["TrackingList"]
        trackingURLsColumn = TrackingList_sheet["F:F"]        
        holdingColumn = TrackingList_sheet["G:G"]
        holdingPercentageColumn = TrackingList_sheet["H:H"]
           
        trackingURLsList = []
        totHoldings = 0
        for indx in range(3, len(trackingURLsColumn)):
            haveIt = False
            if holdingColumn[indx].value == "Hold":
                haveIt = True
                totHoldings += 1
                
            trackingURLsList.append({"URL":trackingURLsColumn[indx].value,
                                     "Hold":haveIt,
                                     "Holding%":holdingPercentageColumn[indx].value})
        
        print(f"# URLs summary: {len(trackingURLsList)} loaded and listed holding {totHoldings}")
        print(trackingURLsList[-5:-1])
        return trackingURLsList
    

#------------------------------------------------------------------------
'''
'''
# https://www.techbeamers.com/handling-html-tables-selenium-webdriver/
# https://chercher.tech/python/table-selenium-python

# globals
column_names = ["date", "fundName", "Quartile", "FERisk", "3m", "6m", "1y", "3y", "5y", "url", "Hold", "Holding%"]
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

    def row_data(self, row_number, *, remove_nl=True):
        if(row_number == 0):
            raise Exception("Row number starts from 1")

        row_number = row_number
        row = self.table.find_elements_by_xpath("//tr["+str(row_number)+"]/td")
        rData = []
        for indx, webElement in enumerate(row) :
            #print(f"[{indx}] |{webElement.text}|")
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
        self.options = Options()
        self.options.binary_location = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
        self.options.add_argument('--disable-site-isolation-trials')
        self.driver = webdriver.Chrome(r"C:\Program Files\Google\Chrome\Application\chromedriver", options = self.options)
        self.sectors_table_page = 1
        #self.driver = webdriver.Chrome()
        self.driver.implicitly_wait(10)


    def click_xpath(self, xpath):
        elem = self.driver.find_element_by_xpath(xpath)
        elem.click()
    
    def click_next(self):
        
        self.sectors_table_page += 1
        self.driver.find_element_by_tag_name('body').send_keys(Keys.END)
        self.driver.implicitly_wait(1)

        #
        
        all_set_page = self.driver.find_elements_by_class_name("set-page")
        for ele in all_set_page:
            if len(ele.text)>0:
                if ele.text == str(self.sectors_table_page):
                    print(f"Found page {self.sectors_table_page} !!")
                    try:

                        element = ele #self.driver.find_element_by_class_name("fe_pagination")
                        #element.location_once_scrolled_into_view
                        #print(element.size['height'])
                        #print(element.location['y'])
                        desired_y = (element.size['height'] / 2) + element.location['y']
                        
                        window_h = self.driver.execute_script('return window.innerHeight')
                        window_y = self.driver.execute_script('return window.pageYOffset')
                        current_y = (window_h / 2) + window_y
                        scroll_y_by = desired_y - current_y
                        
                        for m in [-100]:
                            self.driver.execute_script("window.scrollBy(0, arguments[0]);", scroll_y_by+m)                            
                            time.sleep(1)

                    except Exception as ex:
                        print(f"Failed Next Page {ex}")
                        pass
                    
                    self.driver.implicitly_wait(2)
                    ele.click()
                    self.driver.implicitly_wait(2)
                    return True
                    
        return False
        
    #<label class="form-check-label" for="tc-check-Investor">
    #            <input class="form-check-input tc-check" type="radio" value="Investor" name="tc-check" id="tc-check-Investor">
    #            <span class="custom-checkbox fill-styling"></span>
    #            I am a private investor
    #        </label>
    #<button type="button" class="btn btn-primary mb-3" data-bs-dismiss="modal" id="tc-modal-agree">I agree</button>
    #<span class="custom-checkbox fill-styling"></span>
    def press_I_agree(self):
        print("WebDriverWait")
        WebDriverWait(self.driver, 10).until(lambda d: d.find_element(By.ID, "termsAndConditionsLabel"))
        time.sleep(2)
 
        # I am a private investor
        print("I am a private investor")
        
        # Click on the label instead of the span element
        self.driver.find_element(By.XPATH, "//label[@for='tc-check-Investor']").click()

        # I agree
        print("I agree")

        # Wait for the button to load
        WebDriverWait(self.driver, 10).until(lambda d: d.find_element(By.ID, "tc-modal-agree"))

        #driver.find_element(By.CSS_SELECTOR, "button.btn.btn-primary.mb-3#tc-modal-agree").click()
        # Locate the button using its ID
        agree_button = self.driver.find_element(By.ID, "tc-modal-agree")

        # Click the button
        agree_button.click()                        
        pass
    
    def getFundInf_v2(self, fundUrl, openAndReturn=False):

        if not self._first:
            # open new blank tab
            self.driver.switch_to.window(self.driver.window_handles[0])
            self.driver.execute_script("window.open();")
            time.sleep(5)

            # switch to the new window which is second in window_handles array
            self.driver.switch_to.window(self.driver.window_handles[-1])   

        status = self.driver.get(fundUrl)
        self.driver.implicitly_wait(3)
        print("Get status: ", status)

        if self._first:

            #/html/body/div[1]

            #<button id="CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll" class="CybotCookiebotDialogBodyButton" tabindex="0" lang="en">Allow all</button>
            # Wait for the cookiebot to load
            WebDriverWait(self.driver, 10).until(lambda d: d.find_element(By.ID, "CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll"))

            # Click the 'allow all' button
            allow_all_button = self.driver.find_element(By.ID, "CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll")
            allow_all_button.click()
            
            #<h5 class="modal-title" id="termsAndConditionsLabel">HI GUEST PLEASE TELL US A LITTLE ABOUT YOURSELF SO THAT WE CAN DISPLAY THE MOST APPROPRIATE CONTENT TO YOU:</h5>
            # Wait for the cookiebot to load
            WebDriverWait(self.driver, 10).until(lambda d: d.find_element(By.ID, "termsAndConditionsLabel"))

            # Click the 'allow all' button
            allow_all_button = self.driver.find_element(By.ID, "termsAndConditionsLabel")
            allow_all_button.click()

            self.press_I_agree()
 
            self._first = False

        _statusOK = True
        
        # open web page and return
        if openAndReturn:
            time.sleep(10)
            return _statusOK, self.driver
    
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
                        "Hold": False,
                        "price": "NA",
                        "Holding%": "NA"}

        time.sleep(5)
        try:
            _notFoundTable = True
            #self.driver.implicitly_wait(1)
            #_AllTableElement = self.driver.find_elements_by_class_name("w-100")
            #print(f"_AllTableElement len {len(_AllTableElement)}")
            _AllTableElement = self.driver.find_elements_by_class_name("fe-table")
            #print(f"_AllTableElement len {len(_AllTableElement)}")
            
            for indx, _TableElement in enumerate(_AllTableElement):
                #print(f"[{indx}] |{_TableElement.text}|")

                if re.search('3 m 6 m', _TableElement.text):
                    # found table
                    _notFoundTable = False

                    # get fund name
                    _fundName = self.driver.find_element_by_class_name("key-wrapper__fund-name")
                    fundDict["fundName"] = _fundName.text
                    print(f"fund name {_fundName.text}")

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

            fundDict["Quartile"] = "NA"
            getQuartileRanking = False
            for line in _TableElement.text.split('\n'):
                l += 1

                #print(f"l={l} |{line}| {'3 m 6 m' in line}")
                if '3 m 6 m' in line:
                    _found_3m_6m = True
                    continue
                elif not _found_3m_6m:
                    continue

                if l == 2:
                    valuesList = line.split(' ')
                    for p, key in zip(range(5), ["3m", "6m", "1y", "3y", "5y"]):
                        if is_number(valuesList[p]):
                            fundDict[key] = float(valuesList[p]) 

                if re.search('Quartile Ranking', line):
                    getQuartileRanking = True
                    continue

                if getQuartileRanking:
                    try:
                        fundDict["Quartile"] = int(line)
                    except:
                        pass
                    break
            print(f"\tQuartile={fundDict['Quartile']} fundDict['3m']={fundDict['3m']}")
                

            print("\t>>> Get fund Sector ! ")
            try:
                # look for sector
                # from: https://stackoverflow.com/questions/54862426/python-selenium-get-href-value
                
                #<span class="key-wrapper__fund-name">IT Global Equity Income</span>
                _sector = self.driver.find_elements_by_class_name("key-wrapper__fund-name")
                
                if _sector[1].text.strip():
                    fundDict["Sector"] = _sector[1].text
                else:
                    # empty sector string, use seconder source
                    _sector = self.driver.find_elements_by_xpath(r"/html/body/div[3]/main/section/div/div[1]/div/div[1]/div[1]/p")
                    print(len(_sector), _sector[0].text)
                
                # /html/body/div[3]/main/section/div/div[1]/div/div[1]/div[1]/p/a
                elems = self.driver.find_elements_by_xpath("//div//a[contains(text(),'(View sector)')]")
                fundDict["SectorUrl"] = elems[0].get_attribute("href")

            except Exception as ex:
                print("Sector: ", ex) 

            print("\t\t>>> get price")
            try:
                unit_Information = self.driver.find_element_by_class_name("fe-table.fe_table__head-left.table-all-left")
                unitInformationTable = WebTable(unit_Information)
                price = "NA"
                for line in unitInformationTable.get_text().split('\n'):
                    if not re.search('price', line):
                        continue
                    if re.search('Previous Close price:', line): 
                        fundDict["price"] = line.split(" ")[3]
                    elif re.search('Mid price:', line):
                        fundDict["price"] = line.split(" ")[2]
                    elif re.search('Bid/Offer spread:', line):
                        fundDict["price"] = line.split(" ")[2]
                    else:
                        continue
                    break
                        
                fundDict["price"] = re.subn('[pÂ]', '', fundDict["price"])[0]
                print(f"\t\t>>>>Price is {fundDict['price']}")

            except Exception as ex:
                print("Get price failed: ", ex) 


            print("\t>>> Get Risk ! ")
            #<span class="fe-fundinfo__riskscore">129</span>
            try:
            #     #<span class="risk_score">72</span>
                _FERisk = self.driver.find_element_by_class_name("fe-fundinfo__riskscore")
                fundDict["FERisk"] = int(_FERisk.text)
                print(f"\t\t>>>> Got Risk score {fundDict['FERisk']}! ")
        
            except NoSuchElementException:  
                pass
            
            # success 
            finally:
                
                # close tab 
                # (source: https://medium.com/@pavel.tashev/python-and-selenium-open-focus-and-close-a-new-tab-4cc606b73388)
                self.driver.close()
                time.sleep(1)
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
                
                # _notFoundTable = True
                nextTable = True
                chrom_driver.implicitly_wait(1)
                column_names = ["date", "sectorName", "1m", "3m", "6m", "1y", "3y", "5y"]
                sectors_df = pd.DataFrame(columns = column_names)

                while nextTable:
                    _AllTableElement = chrom_driver.find_elements_by_class_name("table-responsive")
                    for _TableElement in _AllTableElement:
                        

                        if re.search('Name', _TableElement.text):
                            webTable = WebTable(_TableElement)
                            for r in range(webTable.get_row_count()):
                                row_data = webTable.row_data(r+1)
                                while len(row_data)>0 and len(row_data[0])==0:
                                    row_data.pop(0)
                                
                                rowDataList = [current_time ,*row_data[:7]]

                                # append data
                                if len(rowDataList)==8 and len(rowDataList[7])>0:
                                    df_length = len(sectors_df)
                                    sectors_df.loc[df_length] = rowDataList

                            try:
                                nextTable = ChromeInstance.click_next()
                                
                            except Exception as ex:
                                nextTable = False
                            
                                

                # save all funds  information 
                print("# Sectors Information")
                print(sectors_df)
                ## https://chrisalbon.com/python/data_wrangling/pandas_dataframe_importing_csv/
                fileName = "C:\\Users\\tzurv\\projects\\scraper\\" + dateStamp + "_TrustNetSectors.csv"
                print(f"Saving Sectors information to {fileName}")
                sectors_df.to_csv(fileName, sep=',', float_format='%.2f')

                    
            except NoSuchElementException:
                print(f"webpage {url} don't include required performance table")
                _statusOK = False
                
            except Exception as ex:
                print(ex) 
                _statusOK = False

    print("Done !!")

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
        if url == None:
            break
        print(f"URLinf is {URLinf}, totURLs={totURLs}/{len(trackingURLsList)}, totSuccessful={totSuccessful}")
        
        totURLs += 1 
        print(f"# fetching: {url}")
        url = url.rstrip("\n")
        #print(f"# fetching: {url}") 
        
        reTries = 1
        maxNtime = 3
        while reTries<maxNtime:
            
            Status, fundInf = ChromeInstance.getFundInf_v2(url)
            reTries += 1
            
            if Status and not fundInf.empty:
                totSuccessful += 1
                fundInf.loc[0, 'date'] = current_time
                fundInf.loc[0, 'url'] = url
                fundInf.loc[0, 'Hold'] = URLinf['Hold']
                fundInf.loc[0, "Holding%"] =  URLinf["Holding%"]
                
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
    fileName = "C:\\Users\\tzurv\\projects\\scraper\\" + dateStamp + "_FundsInf.csv"
    try:
        print(f"Saving information to {fileName}")    
        allFundsInf.to_csv(fileName, sep=',', float_format='%.2f')
    except:
        print(f"FAILED to save to {fileName}")
        print(f"SAVING TO Local_" + dateStamp + "_FundsInf.csv")    
        allFundsInf.to_csv("Local_" + dateStamp + "_FundsInf.csv", sep=',', float_format='%.2f')

    if not totURLs == totSuccessful:
        print('#'*50)
        print(f"# Not all data collected. File includes {totSuccessful} out of {totURLs} urls in the list #")
        print('#'*50)
        print(failedURLs)

