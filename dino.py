import io
import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
from PIL import Image

game_url = "chrome://dino"
resolution = (256, 256)

class Game():
    def __init__(self):
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    
    def open(self):
        try:
            self.driver.get(game_url)
        except WebDriverException:
            pass

    def get_frame(self):
        frame = self.driver.get_screenshot_as_png()
        return frame

if __name__ == "__main__":
    game = Game()
    game.open()
    frame = game.get_frame()
    img = Image.open(io.BytesIO(frame)).convert("L").resize(resolution)
    img_array = np.array(img)
    print(img_array.shape)
    cv2.imshow("screen", img_array)
    cv2.waitKey()
    