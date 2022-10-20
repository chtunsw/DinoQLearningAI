import io
import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.common.exceptions import WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
from PIL import Image

game_url = "chrome://dino"
browser_position = (0, 0)
browser_size = (512, 256)
frame_resolution = (256, 128)

class Game():
    def __init__(self):
        options = webdriver.ChromeOptions()
        options.add_argument(f"--window-position={browser_position[0]},{browser_position[1]}")
        options.add_argument(f"--window-size={browser_size[0]},{browser_size[1]}")
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    def open(self):
        try:
            self.driver.get(game_url)
        except WebDriverException:
            pass
    
    def start(self):
        self.up()

    def up(self):
        self.driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ARROW_UP)

    def down(self):
        self.driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ARROW_DOWN)

    def get_crashed(self):
        return self.driver.execute_script("return Runner.instance_.crashed")

    def restart(self):
        self.driver.execute_script("Runner.instance_.restart()")

    def get_frame(self):
        screenshot = self.driver.get_screenshot_as_png()
        with Image.open(io.BytesIO(screenshot)) as img:
            img = img.convert("L").resize(frame_resolution)
            frame = np.array(img)
            return frame

    def display(self, frame):
        cv2.imshow("dino screen", frame)
        cv2.moveWindow("dino screen", browser_size[0], browser_position[1])
        cv2.waitKey(1)


if __name__ == "__main__":
    game = Game()
    game.open()
    game.start()
    while True:
        if game.get_crashed():
            game.restart()
        else:
            game.up()
            game.down()
            frame = game.get_frame()
            game.display(frame)