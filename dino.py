import io
import cv2
import base64
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
frame_resolution = (128, 64)
frame_shape = (frame_resolution[1], frame_resolution[0])
action_list = [0, 1, 2]
num_actions = len(action_list)

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
    
    def close(self):
        self.driver.close()
    
    def start(self):
        self.driver.find_element(By.TAG_NAME, "body").send_keys(Keys.SPACE)

    def restart(self):
        self.driver.execute_script("return Runner.instance_.restart()")
    
    def is_crashed(self):
        return self.driver.execute_script("return Runner.instance_.crashed")
    
    def do_nothing(self):
        pass
    
    def down(self):
        self.driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ARROW_DOWN)

    def up(self):
        self.driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ARROW_UP)

    def take_action(self, action):
        if action == 0:
            self.do_nothing()
        elif action == 1:
            self.down()
        elif action == 2:
            self.up()
        next_frame = self.get_frame()
        game_over = self.is_crashed()
        reward = -1 if game_over else 0.1
        return reward, next_frame, game_over

    def get_frame(self):
        image_b64 = self.driver.execute_script("canvasRunner = document.getElementsByClassName('runner-canvas')[0]; \
            return canvasRunner.toDataURL().substring(22)")
        with Image.open(io.BytesIO(base64.b64decode(image_b64))) as img:
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
        if game.is_crashed():
            game.restart()
        game.take_action(0)
        game.take_action(1)
        game.take_action(2)
        frame = game.get_frame()
        game.display(frame)