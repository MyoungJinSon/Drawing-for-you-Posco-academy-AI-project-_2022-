import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import cv2
import os

#for OCR
import requests
import uuid
import time   
import json

#자연어처리-------------
from typing import List
from textrankr import TextRank
import requests
import spacy

#TTScene-----------------
#import _init_paths
import math, cv2, random
import numpy as np
import os.path as osp
from copy import deepcopy
import matplotlib.pyplot as plt

path = '/home/piai/Demo_streamlit/'

#if st.button('take a pic'):

def load_image(image_path):
    img = Image.open(image_path)
    return img

st.title('드로잉 포유📚')

#이미지 보여주기 ----------------------------------------------
#img = load_image(path+'test_demo.png')
img = st.camera_input(" ")
if img is None:
  st.text(" ")
else:
  with open(img.name, "wb") as f:
    f.write(img.getbuffer())

#OCR -------------------------------------------------------
api_url = 'https://8xblx2knfp.apigw.ntruss.com/custom/v1/15124/a68e7898ef33698ad8126367393985e1fb5d7b8a79dbf1d7d66e72dc5b333576/general'
secret_key = 'VklCbWZkWlFXWUZkUUFQVEFuUWdOVXBUcW9UeXdsRmo='

def get_text(image_path):
  image_file = image_path

  request_json = {
      'images': [
          {
              'format': 'jpg',
              'name': 'book'
          }
      ],
      'requestId': str(uuid.uuid4()),
      'version': 'V2',
      'timestamp': int(round(time.time() * 1000))
  }

  payload = {'message': json.dumps(request_json).encode('UTF-8')}
  files = [
    ('file', open(image_file,'rb'))
  ]
  headers = {
    'X-OCR-SECRET': secret_key
  }

  response = requests.request("POST", api_url, headers=headers, data = payload, files = files)

  return json.loads(response.text.encode('utf8'))

#text받아오기
## Buttons 빼도됨
s = ''
s_show = ''
if st.button('START!'):
    #st.text('업로드 되었습니다...')
    ocr_json = get_text(path+img.name)
    #with open('output2.json', 'w', encoding='utf-8') as outfile:
    #    json.dump(ocr_json, outfile, indent=4, ensure_ascii=False)

    for images in ocr_json['images']:
        for key in images['fields']:
            s+=key['inferText'] + " "

    for images in ocr_json['images']:
        for key in images['fields']:
          if key['lineBreak'] == True:
            s_show += key['inferText'] + "\n"
          else:
            s_show += key['inferText'] + " "

    #print(s)


#자연어처리-------------------------------------------

# 동화용 대체 튜플 리스트 ('변경전', '변경후')
replace_word_tuple_list = [('villagers', 'people')]

# papago translate API function
def get_translate(input_text):
    client_id = "rfRvys1WgaW888ughpvm"
    client_secret = "UjTpx8bIkG" 

    data = {'text' : input_text,
            'source' : 'ko',
            'target': 'en'}

    url = "https://openapi.naver.com/v1/papago/n2mt"

    header = {"X-Naver-Client-Id":client_id,
              "X-Naver-Client-Secret":client_secret}

    response = requests.post(url, headers=header, data=data)
    rescode = response.status_code

    if(rescode==200):
        send_data = response.json()
        trans_data = (send_data['message']['result']['translatedText'])
        return trans_data
    else:
        print("Error Code:" , rescode)

# 일반
class MyTokenizer:

  def __call__(self, text: str) -> List[str]:
      tokens: List[str] = text.split()
      return tokens

# summarized function
def summarized(input_text, k=1) :
  mytokenizer: MyTokenizer = MyTokenizer()
  textrank: TextRank = TextRank(mytokenizer)

  summarized: str = textrank.summarize(input_text, k)  

  return summarized

# replace_word
def replace(input_text, replace_word_tuple_list=replace_word_tuple_list):
  
  word_list = input_text.split(" ")  # list화

# [f(x) if condition else g(x) for x in sequence]
  for replace_word in replace_word_tuple_list :

    # 필터링할 단어가 있으면
    if replace_word[0] in word_list :
        word_list = [replace_word[1] if word == replace_word[0] else word for word in word_list] # update

  return " ".join(word_list)

# spacy
def nlp(input_text) :

  # spacy
  _spacy = spacy.load('en_core_web_sm')
  doc = _spacy(input_text)

  filtered = []
  for word in doc:
    # word.lemma_ : -PRON- 같은 케이스 예외처리
    if word.lemma_[0] == '-' :
      filtered.append((word, word, word.dep_))
    else : 
      filtered.append((word, word.lemma_, word.dep_))

  # rule기반 문장 추출
  punct_index = -1
  root_index  = 0 

  for idx,i in enumerate(filtered) :

    if i[2] == 'punct' :
      punct_index = idx
    if i[2] == 'ROOT' :
      root_index  = idx

  if   punct_index == -1 :
    return " ".join([str(word[1]) for word in filtered])

  elif punct_index < root_index :
    return " ".join([str(word[1]) for word in filtered[(punct_index+1):]])

  elif punct_index > root_index :
    return " ".join([str(word[1]) for word in filtered[:punct_index]])

# main
def one_shot(input_text) :
  process_1_output = summarized(input_text)
  process_2_output = get_translate(process_1_output)
  process_3_output = replace(process_2_output)
  process_4_output = nlp(process_3_output)

  return process_4_output

#TTS(speech)---------------------------------------------------------------
# preprocessing
input_text = '<voice name="WOMAN_READ_CALM">  <prosody rate="slow">'+s+' </prosody> </voice>'

# REST_API_KEY
REST_API_KEY = "b9dbc4dea1a2b14b5e766a703eb89f11"

# KakaoTTS
class KakaoTTS:

	def __init__(self, text, API_KEY=REST_API_KEY):
		self.resp = requests.post(
			url="https://kakaoi-newtone-openapi.kakao.com/v1/synthesize",
			headers={
				"Content-Type": "application/xml",
				"Authorization": f"KakaoAK {API_KEY}"
			},
			data=f"<speak>{text}</speak>".encode('utf-8')
		)

	def save(self, filename="output.mp3"):
		with open(filename, "wb") as file:
			file.write(self.resp.content)

try: 
  summary_text = one_shot(s)
  # KakaoTTS 호출
  tts = KakaoTTS(input_text)
  tts.save("output.mp3")
  #play audio file
  audio_file = open(path+"output.mp3", "rb").read()
  st.audio(audio_file, format='audio/mp3', start_time=0)
except:
  st.text('')


#TTS(Scene)-----------------------------------------------------
import sys
sys.path.append('/home/piai/Demo_streamlit/Text2Scene/')

from lib.datasets.abstract_scene import abstract_scene
from lib.modules.abstract_trainer import SupervisedTrainer

from lib.abstract_utils import *
from lib.abstract_config import get_config
import subprocess
if len(s) != 0:
  default_sentence1 = "sun and cloud" 
  default_sentence2 = "glasses and sunglasses" 
  #st.text(summary_text)
  #example_text = "a shepherd boy lives in town"
  sentence_list = [default_sentence1, default_sentence2, summary_text]
  cnt = 0
  with open("sen_json.json",'w') as json_file:
    json.dump([sentence_list],json_file)

  if __name__ == '__main__':
    cnt += 1
    subprocess.call("/home/piai/Demo_streamlit/Text2Scene/tools/abstract_demo.py",shell=True)

  scene_created = load_image(path+'Text2Scene/logs/scene_created/000000000.jpg')

  col1, col2 = st.columns(2)

  with col1:
    st.text(s_show)
  with col2:
    st.image(scene_created, width=400)
else:
  st.text('')