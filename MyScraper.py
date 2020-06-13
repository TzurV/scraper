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

import chromedriver_binary # Adds chromedriver binary to path

import pandas as pd
from datetime import datetime

import sys
#============================================
# (Scraper) C:\Users\...\scraper>pip freeze
# astroid==2.4.0
# certifi==2020.4.5.1
# cffi==1.14.0
# chromedriver-binary==83.0.4103.39.0
# colorama==0.4.3
# cryptography==2.9.2
# idna==2.9
# isort==4.3.21
# lazy-object-proxy==1.4.3
# mccabe==0.6.1
# numpy==1.18.5
# pandas==1.0.4
# pycparser==2.20
# pylint==2.5.0
# pyOpenSSL==19.1.0
# PySocks==1.7.1
# python-dateutil==2.8.1
# pytz==2020.1
# selenium==3.141.0
# six==1.14.0
# toml==0.10.0
# typed-ast==1.4.1
# urllib3==1.25.8
# win-inet-pton==1.1.0
# wincertstore==0.2
# wrapt==1.11.2

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

import unittest

#driver = webdriver.Chrome()
#driver.get("https://www.trustnet.com/factsheets/o/a6x2/first-state-global-listed-infrastructure")


# https://www.techbeamers.com/handling-html-tables-selenium-webdriver/
# https://chercher.tech/python/table-selenium-python

# globals
column_names = ["date", "fundName", "Quartile", "FERisk", "3m", "6m", "1y", "3y", "5y"]
Empty_fund_df = pd.DataFrame(columns = column_names)


#--------------------------------
#from selenium import webdriver
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

        row_number = row_number + 1
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
            A = self.table.find_element_by_xpath("//tr["+str(row_number)+"]/td["+str(column_number)+"]").text
    
        except NoSuchElementException:  #spelling error making this code not work as expected
            return False

        return True

    def get_cell_data(self, row_number, column_number):
        if(row_number == 0):
            raise Exception("Row number starts from 1")

        row_number = row_number + 1
        cellData = self.table.find_element_by_xpath("//tr["+str(row_number)+"]/td["+str(column_number)+"]").text
        return cellData

class Test(unittest.TestCase):
    def test_web_table(self):
        #driver = webdriver.Chrome(executable_path=r'D:/PATH/chromedriver.exe')
        driver = webdriver.Chrome()
        driver.implicitly_wait(30)

        #/html/body/div[1]/div/div[1]/div/button[2]


        driver.get("https://chercher.tech/practice/table")
        w = WebTable(driver.find_element_by_xpath("//table[@id='webtable']"))

        print("No of rows : ", w.get_row_count())
        print("------------------------------------")
        print("No of cols : ", w.get_column_count())
        print("------------------------------------")
        print("Table size : ", w.get_table_size())
        print("------------------------------------")
        print("First row data : ", w.row_data(1))
        print("------------------------------------")
        print("First column data : ", w.column_data(1))
        print("------------------------------------")
        print("All table data : ", w.get_all_data())
        print("------------------------------------")
        print("presence of data : ", w.presence_of_data("Chercher.tech"))
        print("------------------------------------")
        print("Get data from Cell : ", w.get_cell_data(2, 2))

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

    def getFundInf(self, fundUrl):
        status = self.driver.get(fundUrl)
        self.driver.implicitly_wait(10)
        print("Get status: ", status)

        if self._first :
            input("\n ------------------ \n >> Set Agree options befor provessing: \n ")
            self._first = False

        _statusOK = True
        fundInf = Empty_fund_df.copy()
 
        try:
            TableXpath = "/html/body/div[1]/div[2]/div[1]/div/fund-factsheet/section/div[2]/fund-tabs/div/div/fund-tab[1]/div/overview/div/div[1]/div[2]/div[1]/div/div[1]/cumulative-performance"
            w1 = WebTable(self.driver.find_element_by_xpath(TableXpath))
            #print(w1.get_all_data())
            #row_number = 2 - 1 # get_cell_data has 'row_number = row_number + 1'
            #for column_number in range(2,7):
            #    print(w1.get_cell_data(row_number, column_number))

        except NoSuchElementException:
            print(f"webpage {fundUrl} don't include required performance table")
            _statusOK = False
            
        except Exception as ex:
            #template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            #message = template.format(type(ex).__name__, ex.args)
            #print(message)
            #print(type(ex))    # the exception instance
            #print(ex.args)     # arguments stored in .args
            print(ex) 
            _statusOK = False

        if not _statusOK:
            return False, fundInf

        # get performance 
        try:
            fundDict =  {   "date":"NA",
                            "fundName": "NA",
                            "3m": "NA",
                            "6m": "NA",
                            "1y": "NA",
                            "3y": "NA",
                            "5y": "NA",
                            "Quartile": "NA",
                            "FERisk": "NA"}

            #<span ng-if="!perfData.isSector &amp;&amp; perfData.name !== 'Position'" class="fundName bold_text">JPM Asia Growth C Acc</span>
            '/html/body/div[1]/div[2]/div[1]/div/fund-factsheet/section/div[2]/fund-tabs/div/div/fund-tab[1]/div/overview/div/div[1]/div[2]/div[1]/div/div[1]/cumulative-performance/div[1]/performance-table/div[1]/span[2]'
            _fundName = self.driver.find_element_by_class_name("fundName")
            fundDict["fundName"] = _fundName.text

            row_number = 2 - 1 # get_cell_data has 'row_number = row_number + 1'
            fundDict["3m"] = float(w1.get_cell_data(row_number, 2))
            fundDict["6m"] = float(w1.get_cell_data(row_number, 3))
            fundDict["1y"] = float(w1.get_cell_data(row_number, 4))

            #if w1.check_cell_data(row_number, 5):
            _3y = w1.get_cell_data(row_number, 4)
            if is_number(_3y):
                fundDict["3y"] = float(_3y)

            #if w1.check_cell_data(row_number, 6):
            _5y = w1.get_cell_data(row_number, 6)
            if is_number(_5y):
                fundDict["5y"] = float(_5y)

            if w1.check_cell_data(4,2):
                fundDict["Quartile"] = int(w1.get_cell_data(4, 2))
            
            # https://www.selenium.dev/selenium/docs/api/py/webdriver_remote/selenium.webdriver.remote.webelement.html

        except Exception as ex:
            print(f"Failed to get performance for {fundUrl}")
            print(ex) 
            return False, fundInf

        try:
            #<span class="risk_score">72</span>
            _FERisk = self.driver.find_element_by_class_name("risk_score")
            fundDict["FERisk"] = int(_FERisk.text)
    
        except NoSuchElementException:  #spelling error making this code not work as expected
            pass

        # add the data
        print(fundDict)
        
    
#        return True, fundInf.append(fundDict, ignore_index=True)
        return True, pd.DataFrame(fundDict, index=[0])



if __name__ == "__main__":
    #unittest.main()


    #driver = webdriver.Chrome()
    #driver.implicitly_wait(30)

    #driver.get("https://chercher.tech/practice/table")
    #driver.implicitly_wait(30)

    #w = WebTable(driver.find_element_by_xpath("//table[@id='webtable']"))
    #print(w.get_table_size())

    #driver.get("https://www.trustnet.com/factsheets/o/be80/baillie-gifford-pacific-b-acc")
    #w = WebTable(driver.find_element_by_xpath("//table"))
    #print(x)

    #w1 = WebTable(driver.find_element_by_xpath("/html/body/div[1]/div[2]/div[1]/div/fund-factsheet/section/div[2]/fund-tabs/div/div/fund-tab[1]/div/overview/div/div[1]/div[2]/div[1]/div/div[1]/cumulative-performance/div[1]/performance-table/table"))
    #print(w1.get_table_size())
    #print(w1.get_all_data())
    #print("Done!")

    #----------------------
    # Get current time date  
    #https://docs.python.org/3/library/datetime.html
    now = datetime.now()

    current_time = now.strftime("%d/%m/%y %H:%M")
    print("Current Time =", current_time)

    # collect all funds information
    allFundsInf = Empty_fund_df.copy()


    #============
    # start chrom
    
    ChromeInstance = trustnetInf()

    # dev  case for 2 funds
    if True:
        # test 
        #   https://www.trustnet.com/factsheets/o/k5lq/fidelity-global-health-care
        #   https://www.trustnet.com/factsheets/o/ngpb/baillie-gifford-positive-change-b-acc
        #   https://www.trustnet.com/factsheets/o/nbh5/lindsell-train-global-equity-b-gbp

        url = "https://www.trustnet.com/factsheets/o/be80/baillie-gifford-pacific-b-acc"
        Status, fundInf = ChromeInstance.getFundInf(url)
        print(Status)
        print(fundInf)
        if Status and not fundInf.empty:
            fundInf.loc[0, 'date'] = current_time
            fundInf.loc[0, 'url'] = url
            allFundsInf = allFundsInf.append(fundInf, ignore_index=True)
            #allFundsInf = allFundsInf_tmp.copy()

        url = "https://www.trustnet.com/factsheets/o/ngpb/baillie-gifford-positive-change-b-acc"
        Status, fundInf = ChromeInstance.getFundInf(url)
        print(Status)
        print(fundInf)
        if Status and not fundInf.empty:
            fundInf.loc[0, 'date'] = current_time
            fundInf.loc[0, 'url'] = url
            allFundsInf = allFundsInf.append(fundInf, ignore_index=True)

        print(allFundsInf)

        ## https://chrisalbon.com/python/data_wrangling/pandas_dataframe_importing_csv/
        allFundsInf.to_csv("C:\\Users\\tzurv\\python\\VScode\\scraper\\DevFundsInf.csv", sep='\t', float_format='%.2f')

        allFundsInf_Verify = pd.read_csv('C:\\Users\\tzurv\\python\\VScode\\scraper\\DevFundsInf.csv', sep='\t')
        print(allFundsInf_Verify)

        print("exit Main!")
        sys.exit(0)
 
    # loop over a list in a file
    with open('FundsUrls.txt', 'r') as fh:
        for url in fh:
            url = url.rstrip("\n")
            print(url)       
            Status, fundInf = ChromeInstance.getFundInf(url)
            #print(fundInf)
            if Status and not fundInf.empty:
                fundInf.loc[0, 'date'] = current_time
                fundInf.loc[0, 'url'] = url
                allFundsInf = allFundsInf.append(fundInf, ignore_index=True)
                print(allFundsInf)

    # save all funds  information 
    ## https://chrisalbon.com/python/data_wrangling/pandas_dataframe_importing_csv/
    allFundsInf.to_csv("C:\\Users\\tzurv\\python\\VScode\\scraper\\FundsInf.csv", sep='\t', float_format='%.2f')







