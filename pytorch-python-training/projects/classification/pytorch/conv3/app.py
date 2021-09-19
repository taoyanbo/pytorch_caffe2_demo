from flask import Flask
from flask import json
from flask import request

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import time
import os
from PIL import Image
import sys
import torch.nn.functional as F

from net import simpleconv3

app = Flask(__name__)
 

@app.route("/")
def index():
    return "Index!"
 
@app.route("/hello")
def hello():
    return "Hello World!"
 
@app.route("/members")
def members():
    return "Members"
 
@app.route("/members/<string:name>/")
def getMember(name):
    return name
	
@app.route('/upload', methods=['POST'])
def upload():
	if request.headers['Content-Type'] == 'image/bmp':
		print (request.headers)
		with open("./image.bmp", 'wb') as f:
			f.write(request.data)
			f.close()
		
		data_transforms =  transforms.Compose([
            transforms.Resize(48),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])

		net = simpleconv3()
		modelpath = './models/model.ckpt';
		net.load_state_dict(torch.load(modelpath,map_location=lambda storage,loc: storage))
		imagepath = "./image.bmp";
		image = Image.open(imagepath)
		imgblob = data_transforms(image).unsqueeze(0)
		imgblob = Variable(imgblob)

		torch.no_grad()

		predict = F.softmax(net(imgblob))
		print(predict.data.stride())
		return '%.5f, %.5f,%.5f' % (predict.data[0][0].item(),predict.data[0][1].item(),predict.data[0][2].item())
	else:
		return "BMPFAIL"
	
if __name__ == "__main__":
    app.run()