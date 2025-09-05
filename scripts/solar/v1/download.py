import time
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def pipeline_solar_explorator(coords, download_dir, sleep_time=5):
    options = Options()
    options.add_experimental_option("prefs", {
        "download.default_directory": download_dir,
        "download.prompt_for_download": False,
        "safebrowsing.enabled": True
    })
    driver = webdriver.Chrome(options=options)
    wait = WebDriverWait(driver, 20)

    driver.get("https://solar.minenergia.cl/exploracion")


    def steps():
        # Input lat
        lat_input = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="buscaSitioLat"]')))
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", lat_input)
        lat_input.clear()
        lat_input.send_keys(str(lat))

        # Input lon
        lon_input = driver.find_element(By.XPATH, '//*[@id="buscaSitioLon"]')
        lon_input.clear()
        lon_input.send_keys(str(lon))

        time.sleep(2) 
        # Botón buscar
        safe_click_xpath("/html/body/main/div/div/div/div[3]/div[2]/div[1]/div/div[3]/button", "botón buscar")
        time.sleep(sleep_time)  # esperar a que cargue el modal

        # Pestaña descargas
        safe_click_xpath("/html/body/main/div/div/div/div[6]/div[2]/div/div/div/div/div[3]/button/div", "pestaña descargas")
        time.sleep(2)

        # Botón de descarga
        safe_click_xpath("/html/body/main/div/div/div/div[6]/div[2]/div/div/descargas-exploracion/div/div/table/tbody/tr[1]/td[3]/button", "botón descarga")
        time.sleep(sleep_time+2)

    def safe_click_xpath(xpath, label, retries=3):
        for i in range(retries):
            try:
                elem = wait.until(EC.presence_of_element_located((By.XPATH, xpath)))
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", elem)
                time.sleep(0.3)
                driver.execute_script("arguments[0].click();", elem)
                return
            except Exception as e:
                print(f"❌ [{i+1}/{retries}] Error al hacer clic en {label}: {e}")
                time.sleep(1)
        raise Exception(f"No se pudo hacer clic en {label} tras {retries} intentos.")

    # Botón de búsqueda (imagen lupa)
    safe_click_xpath("/html/body/main/div/div/div/div[3]/div[1]/div[2]/img", "botón búsqueda")
    time.sleep(sleep_time - 2)

    for lat, lon in tqdm(coords, desc="Procesando coordenadas"):
        try:
            print(f'lat, lon ({lat}, {lon})')
            steps()
            # Volver al botón de búsqueda
            safe_click_xpath("/html/body/main/div/div/div/div[3]/div[1]/div[2]/img", "botón búsqueda (reinicio)")

        except Exception as e:
            print(f"⚠️ Error en ({lat}, {lon}): {e}")
            driver.save_screenshot(f"error_{lat}_{lon}.png")
            time.sleep(sleep_time)
            try:  
                safe_click_xpath("/html/body/main/div/div/div/div[3]/div[1]/div[2]/img", "botón búsqueda (reinicio)")
                steps()

            except Exception as e:
                print(f"⚠️ Error!!! en ({lat}, {lon}): {e}")
                driver.save_screenshot(f"error_{lat}_{lon}.png")
                time.sleep(sleep_time)               
    driver.quit()
