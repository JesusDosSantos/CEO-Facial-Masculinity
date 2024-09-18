from concurrent.futures import ThreadPoolExecutor
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import os
import requests
import time
import base64
from PIL import Image
from io import BytesIO
#import beatifulsoup4
import bs4

N_THREADS = 30 #adjust this to your liking
N_IMAGES = 15
main_download_dir = "downloaded_images test"
url_data = {}


def download_ceo_images(ceo_list):
    ceo = ceo_list[0]
    company = ceo_list[1]
    year_range = ceo_list[3]
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    local_driver = webdriver.Chrome(options=options)
    local_driver.get("https://www.google.com/imghp")
    search_query = f"{ceo} Headshot {company}"

    search_input = local_driver.find_element(By.NAME, "q")
    search_input.send_keys(search_query)
    search_input.send_keys(Keys.RETURN)

    time.sleep(2)
    image_elements = local_driver.find_elements(By.TAG_NAME, "img")
    sorted_images = []

    for i, img_elem in enumerate(image_elements):
        if i < 100:
            try:
                width = int(img_elem.get_attribute("width"))
                height = int(img_elem.get_attribute("height"))

                if width > 50 and height > 50:
                    sorted_images.append(img_elem)
            except Exception as e:
                print(f"[{ceo}] Error processing image {i}: {e}")

    # Create a directory for the company first
    company_dir = os.path.join(main_download_dir, company)
    os.makedirs(company_dir, exist_ok=True)

    # Inside the company directory, create a directory for the CEO with the year range
    ceo_year_dir = os.path.join(company_dir, f"{ceo}_{year_range}")
    os.makedirs(ceo_year_dir, exist_ok=True)

    session = requests.Session()
    ceo_key = (ceo, company, year_range)  # This will uniquely identify each CEO entry

    if ceo_key not in url_data:
        url_data[ceo_key] = []

    for i, img_elem in enumerate(sorted_images):
        if i < N_IMAGES:
            try:
                img_elem.click()

                img_element = WebDriverWait(local_driver, 3).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'img[jsname="kn3ccd"]'))
                )
                img_url = img_element.get_attribute("src")
                url_data[ceo_key].append(img_url)

                # Save the images inside the ceo_year_dir
                if img_url.startswith("data:image"):
                    img_data = base64.b64decode(img_url.split(",")[1])
                    img = Image.open(BytesIO(img_data))
                    img = img.convert("RGB")
                    img.save(os.path.join(ceo_year_dir, f"image_{i + 1}.jpg"), "JPEG")

                else:
                    img_data = session.get(img_url).content
                    with open(os.path.join(ceo_year_dir, f"image_{i + 1}.jpg"), "wb") as img_file:
                        img_file.write(img_data)

            except Exception as e:
                print(f"[{ceo}] Error processing image {i}: {e}")

    local_driver.quit()

if __name__ == "__main__":
    df = pd.read_csv('ceo_cfo.csv')

    df = df[df['is_ceo'] == 'Y']

    df['exec_lname'] = df['exec_lname'].str.split(',').str[0]

    df['full_name'] = df['exec_fname'] + " " + df['exec_lname']
    grouped = df.groupby(['full_name', 'coname'])['year'].agg(['min', 'max'])

    grouped['years'] = grouped['min'].astype(str) + "-" + grouped['max'].astype(str)
    result_df = grouped.reset_index()[['full_name', 'coname', 'min', 'years']]

    result_df = result_df.sort_values(['coname', 'min'])

    result_df = result_df.head(100)

    CEOs = result_df.values.tolist()

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
        executor.map(download_ceo_images, CEOs)

    # Merge url_data with your dataframe
    df_url = pd.DataFrame(list(url_data.keys()), columns=['full_name', 'coname', 'years'])
    df_url['image_urls'] = [';'.join(urls) for urls in url_data.values()]

    df_merged = pd.merge(result_df, df_url, on=['full_name', 'coname', 'years'], how='left')
    df_merged.to_csv('ceo_cfo_with_URLs.csv', index=False)

    print("--- %s seconds ---" % (time.time() - start_time))
