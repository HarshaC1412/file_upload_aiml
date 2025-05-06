import time
import subprocess
import random
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import os

# Directory for screenshots
SCREENSHOT_DIR = "screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# Start the Flask application
print("Starting Flask application...")
flask_process = subprocess.Popen(["python", "app.py"])

# Wait for Flask app to start and verify
time.sleep(10)
print("Checking if Flask app is running...")
for _ in range(5):
    try:
        response = requests.get("http://127.0.0.1:5000/", timeout=2)
        if response.status_code == 200:
            print("Flask app is running.")
            break
    except requests.ConnectionError:
        print("Waiting for Flask app to start...")
        time.sleep(2)
else:
    raise Exception("Flask app failed to start within timeout.")

# Initialize WebDriver (Chrome)
driver = webdriver.Chrome()
wait = WebDriverWait(driver, 15)  # Timeout 15s
driver.maximize_window()

def take_screenshot(step_name):
    """Capture a screenshot with timestamp."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    screenshot_path = os.path.join(SCREENSHOT_DIR, f"{step_name}_{timestamp}.png")
    driver.save_screenshot(screenshot_path)
    print(f"Screenshot saved: {screenshot_path}")

try:
    # Step 1: Navigate to home page and login
    print("Step 1: Navigating to home page and logging in...")
    driver.get("http://127.0.0.1:5000/")
    wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    print(f"Home page title: '{driver.title}'")
    take_screenshot("home_page")
    time.sleep(2)

    print("Clicking Login link...")
    login_link = driver.find_element(By.LINK_TEXT, "Login")
    login_link.click()
    wait.until(EC.presence_of_element_located((By.NAME, "username")))
    print("Login page loaded successfully.")
    take_screenshot("login_page")
    time.sleep(2)

    print("Filling login form...")
    username_input = driver.find_element(By.NAME, "username")
    password_input = driver.find_element(By.NAME, "password")
    
    print("Entering username: 'harsha_c'")
    username_input.send_keys("harsha_c")
    time.sleep(1)
    print("Entering password: 'harsha1412'")
    password_input.send_keys("harsha1412")
    time.sleep(1)

    print("Clicking login submit button...")
    try:
        submit_button = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
    except:
        print("Button not found, trying input[type='submit']...")
        submit_button = driver.find_element(By.CSS_SELECTOR, "input[type='submit']")
    submit_button.click()
    wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    print(f"Index page title: '{driver.title}'")
    print("Login successful, redirected to index page.")
    take_screenshot("index_page")
    time.sleep(2)

    # Step 2: Upload files
    print("Step 2: Uploading files...")
    file_inputs = {
        "l_file": driver.find_element(By.NAME, "l_file"),
        "s_file": driver.find_element(By.NAME, "s_file"),
        "c_file": driver.find_element(By.NAME, "c_file"),
        "h_file": driver.find_element(By.NAME, "h_file"),
        "i_file": driver.find_element(By.NAME, "i_file")
    }

    file_paths = {
        "l_file": "C:/Users/Harsha/Documents/LandFile.csv",
        "s_file": "C:/Users/Harsha/Documents/SalesFile.csv",
        "c_file": "C:/Users/Harsha/Documents/CurrentValue.csv",
        "h_file": "C:/Users/Harsha/Documents/HistoricalFile.csv",
        "i_file": "C:/Users/Harsha/Documents/ImprovementFile.csv"
    }

    for key, path in file_paths.items():
        print(f"Uploading {key}: {path}")
        file_inputs[key].send_keys(path)
        time.sleep(1)
    take_screenshot("files_uploaded")
    time.sleep(2)

    # Step 3: Fill month filters and county name
    print("Step 3: Filling month filters and county name...")
    res_months = 36
    com_months = 48
    vl_months = 60
    county_name = "Default_County"

    driver.find_element(By.NAME, "res").send_keys(str(res_months))
    print(f"Set Residential months: {res_months}")
    time.sleep(1)
    driver.find_element(By.NAME, "com").send_keys(str(com_months))
    print(f"Set Commercial months: {com_months}")
    time.sleep(1)
    driver.find_element(By.NAME, "vl").send_keys(str(vl_months))
    print(f"Set Vacant Land months: {vl_months}")
    time.sleep(1)
    driver.find_element(By.NAME, "cname").send_keys(county_name)
    print(f"Set County name: {county_name}")
    take_screenshot("filters_filled")
    time.sleep(2)

    # Step 4: Click Validate Data button and handle popup
    print("Step 4: Clicking Validate Data button...")
    try:
        validate_button = driver.find_element(By.ID, "validate_data")
        print("Found Validate Data button by ID: validate_data")
    except:
        print("ID 'validate_data' not found, trying button with text 'Validate Data'...")
        validate_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Validate Data')]")
    validate_button.click()
    print("Validate Data button clicked.")
    take_screenshot("validate_clicked")
    time.sleep(5)  # Delay for popup to load

    print("Waiting for popup...")
    popup = wait.until(EC.visibility_of_element_located((By.ID, "successModal")))
    print("Popup found with ID 'successModal'.")
    take_screenshot("popup_loaded")

    print("Checking for success message...")
    popup_text = popup.text.lower()
    print("Popup full text:", popup_text)
    if "success" in popup_text:
        print("Success message confirmed in popup content.")
    else:
        print("No 'success' text found, dumping popup HTML for debugging...")
        print("Popup HTML:", popup.get_attribute("outerHTML"))
        raise Exception("Validation failed: No success message found in popup content.")

    print("Attempting to close popup...")
    close_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//div[@id='successModal']//button[@class='close']")))
    print("Close button found with XPath: //div[@id='successModal']//button[@class='close']")
    close_button.click()
    wait.until(EC.invisibility_of_element_located((By.ID, "successModal")))
    print("Popup closed successfully.")
    take_screenshot("popup_closed")
    time.sleep(2)

    # Step 5: Click Run Audit Checker button
    print("Step 5: Clicking Run Audit Checker button...")
    try:
        # Wait for the "Run Audit Checker" button to be clickable
        audit_button = wait.until(EC.element_to_be_clickable((By.ID, "runAuditBtn")))
        print("Found Run Audit Checker button by ID: runAuditBtn")
        
        # Click the "Run Audit Checker" button
        audit_button.click()
        print("Run Audit Checker button clicked.")
        take_screenshot("audit_clicked")
        
        # Wait for the form submission to complete and the page to redirect to the results page
        wait.until(EC.url_contains("/results"))
        print(f"Current URL: {driver.current_url}")
        print("Form submitted successfully. Redirected to results page.")
        take_screenshot("results_page")
        time.sleep(2)

        # Step 6: Wait for the results to be processed and redirected to the report page
        print("Step 6: Waiting for report generation and redirection...")
        wait.until(EC.url_contains("/report/"))
        print(f"Current URL: {driver.current_url}")
        print("Redirected to report page.")
        take_screenshot("report_page")
        time.sleep(2)

    except Exception as e:
        print(f"Error during Step 5 or Step 6: {str(e)}")
        take_screenshot("error_state")
        raise e

    # Step 6: Scroll report page and download PDF
    print("Step 6: Scrolling report page and downloading PDF...")

    # Wait for the report page to fully load
    wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    print("Report page fully loaded.")

    # Scroll to the bottom of the page
    print("Scrolling to the bottom of the report page...")
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)  # Wait for scrolling to complete
    take_screenshot("report_bottom")

    # Scroll back to the top of the page
    print("Scrolling back to the top of the report page...")
    driver.execute_script("window.scrollTo(0, 0);")
    time.sleep(2)  # Wait for scrolling to complete
    take_screenshot("report_top")

    # Locate and click the "Download Report" button
    print("Clicking Download Report button...")
    try:
        # Wait for the download button to be clickable
        download_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//a[contains(@href, '/download_pdf')]")))
        print(f"Download button found with href: {download_button.get_attribute('href')}")
        
        # Click the download button
        download_button.click()
        print("PDF download initiated.")
        take_screenshot("download_initiated")
        
        # Wait for the download to complete (adjust time as needed)
        time.sleep(5)  # Allow time for the file to download
    except Exception as e:
        print(f"Error during Step 6: {str(e)}")
        take_screenshot("error_state")
        raise e

    # Pause for inspection
    print("Testing completed. Browser will remain open for 30 seconds...")
    time.sleep(30)

except Exception as e:
    print(f"Error during testing: {str(e)}")
    take_screenshot("error_state")

finally:
    print("Step 7: Cleaning up...")
    driver.quit()  # Close browser
    flask_process.terminate()
    print("Flask process terminated. Testing fully completed.")