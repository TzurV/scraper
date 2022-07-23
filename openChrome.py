# https://thewebdev.info/2022/04/16/how-to-fix-webdriverexception-unknown-error-cannot-find-chrome-binary-error-with-selenium-in-python-for-older-versions-of-google-chrome/
# https://developercommunity.visualstudio.com/t/selenium-ui-test-can-no-longer-find-chrome-binary/1170486

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time


options = Options()
options.binary_location = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
driver = webdriver.Chrome(r"C:\Program Files\Google\Chrome\Application\chromedriver", chrome_options = options)
driver.get('https://www.trustnet.com/')
time.sleep(5)
driver.quit()
