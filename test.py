from __future__ import print_function
import argparse
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms
import utils
from model import reflection
# Training settings
parser = argparse.ArgumentParser(description='parser for Reflection Removal')
parser.add_argument('--input', type=str, required=True, help='input image folder to use')
parser.add_argument('--model', type=str, required=True, help='model file to use')
parser.add_argument('--cuda', action='store_true', help='use cuda')
args = parser.parse_args()
# load model
model = reflection()
checkpoint = torch.load(args.model)
model.load_state_dict(checkpoint['state_dict_G'])
if args.cuda:
    model.cuda()
model.eval()

with torch.no_grad():
    input = utils.load_image(args.input)
    width, height= input.size
    input = input.crop((0,0, width-width%4, height-height%4))
    input_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    input = input_transform(input)
    input_mean = torch.mean(torch.mean(input, 1), 1)
    input_mean_tensor = input.new(input.size())
    input_mean_tensor[0, :, :] = input_mean[0]
    input_mean_tensor[1, :, :] = input_mean[1]
    input_mean_tensor[2, :, :] = input_mean[2]
    input = input.unsqueeze(0)
    input_mean_tensor = input_mean_tensor.unsqueeze(0)
    if args.cuda:
        input = input.cuda()
        input_mean_tensor = input_mean_tensor.cuda()

    transmit = model(input)
    reflected = input-transmit
    reflected = reflected + input_mean_tensor

    # normalization
    for cc in range(0,3):
        numerator = torch.dot(transmit[:,cc,:,:].view(-1), input[:,cc,:,:].view(-1))
        denominator = torch.dot(transmit[:,cc,:,:].view(-1), transmit[:,cc,:,:].view(-1))
        alpha = numerator / denominator
        transmit[:,cc,:,:] = transmit[:,cc,:,:] * alpha

        numerator = torch.dot(reflected[:,cc,:,:].view(-1), input[:,cc,:,:].view(-1))
        denominator = torch.dot(reflected[:,cc,:,:].view(-1), reflected[:,cc,:,:].view(-1))
        alpha = numerator / denominator
        reflected[:,cc,:,:] = reflected[:,cc,:,:] * alpha


    if args.cuda:
        output1 = transmit.cpu()
        output2 = reflected.cpu()

    output1_data = output1.data[0]*255
    utils.save_image('background.png', output1_data)
    output2_data = output2.data[0]*255
    utils.save_image('reflection.png', output2_data)
