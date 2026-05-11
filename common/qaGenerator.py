from openai import OpenAI
import configparser

conf = configparser.ConfigParser()
conf.read("./config.ini")

API_KEY = conf.get("API_KEY", "QWenAPIKey")

client = OpenAI(api_key=API_KEY, 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)