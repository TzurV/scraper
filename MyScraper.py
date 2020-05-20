# my source is https://www.pluralsight.com/guides/guide-scraping-dynamic-web-pages-python-selenium


#import chromedriver-binary
#path = cdi.install(file_directory='c:\\data\\chromedriver\\', verbose=True, chmod=True, overwrite=False, version=None)
#print('Installed chromedriver to path: %s' % path)

# coppied from
# https://pypi.org/project/chromedriver-binary/
# https://www.youtube.com/watch?v=FFDDN1C1MEQ - Learn Selenium Python
from selenium import webdriver
import chromedriver_binary # Adds chromedriver binary to path

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

if __name__ == "__main__":
    #unittest.main()


    driver = webdriver.Chrome()
    driver.implicitly_wait(30)

 
    #driver.get("https://chercher.tech/practice/table")
    #driver.implicitly_wait(30)

    #w = WebTable(driver.find_element_by_xpath("//table[@id='webtable']"))
    #print(w.get_table_size())

    driver.get("https://www.trustnet.com/factsheets/o/be80/baillie-gifford-pacific-b-acc")
    #w = WebTable(driver.find_element_by_xpath("//table"))
    #print(x)
    w1 = WebTable(driver.find_element_by_xpath("/html/body/div[1]/div[2]/div[1]/div/fund-factsheet/section/div[2]/fund-tabs/div/div/fund-tab[1]/div/overview/div/div[1]/div[2]/div[1]/div/div[1]/cumulative-performance/div[1]/performance-table/table"))
    print(w1.get_table_size())
    print(w1.get_all_data())
    print("Done!")


    #/html/body/div[1]/div[2]/div[1]/div/fund-factsheet/section/div[2]/fund-tabs/div/div/fund-tab[1]/div/overview/div/div[1]/div[2]/div[1]/div/div[1]/cumulative-performance/div[1]/performance-table/table/tbody/tr[2]/td[2]