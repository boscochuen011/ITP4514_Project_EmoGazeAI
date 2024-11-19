import os
import requests
import logging
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

input_image = input("Get photo：")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


webdriver_service = Service(ChromeDriverManager().install())


webdriver_options = Options()
prefs = {"profile.managed_default_content_settings.images": 2}
webdriver_options.add_experimental_option("prefs", prefs)


driver = webdriver.Chrome(service=webdriver_service, options=webdriver_options)

driver.get(f'https://unsplash.com/s/photos/{input_image}')

html = driver.page_source

driver.quit()

soup = BeautifulSoup(html, 'html.parser')

img_tags = soup.find_all('img')

current_path = os.path.dirname(os.path.abspath(__file__))
images_folder = os.path.join(current_path, 'images')

if not os.path.exists(images_folder):
    os.makedirs(images_folder)

img_urls = [img['src'] for img in img_tags if 'src' in img.attrs and img['src'].startswith(('http', 'https'))]

for url in img_urls:
    response = requests.get(url)
    filename = url.split('/')[-1]
    with open(os.path.join(images_folder, filename), 'wb') as f:
        f.write(response.content)
    logging.info('Downloaded %s', filename)
    
    
import os
import glob

images_folder = os.path.join(current_path, 'images')

files = glob.glob(os.path.join(images_folder, '*'))

for filename in files:
    if not filename.lower().endswith('.jpg'):
        new_filename = os.path.splitext(filename)[0] + '.jpg'
        os.rename(filename, new_filename)
        logging.info('Renamed %s to %s', filename, new_filename)