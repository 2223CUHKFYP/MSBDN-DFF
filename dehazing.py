# 17/02/2023 this is for dehazing


# from __future__ import print_function
import argparse
import time
from math import log10
from os.path import join
from torchvision import transforms
from torchvision import utils as utils
import torch
from torch.utils.data import DataLoader
from datasets.dataset_hf5 import DataValSet
import statistics
import re
import torch.nn as nn
import torch.nn.functional as F
import pytorch_ssim
from tqdm import tqdm
import os
import glob


class bcolors:
    # https://gist.github.com/tuvokki/14deb97bef6df9bc6553
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    YELLOW = '\033[93m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def colored(message, color):
        return color + message + bcolors.ENDC

    # Method that returns a yellow warning
    # usage:
    #   print(bcolors.warning("What you are about to do is potentially dangerous. Continue?"))
    @staticmethod
    def warning(message):
        return bcolors.WARNING + message + bcolors.ENDC

    # Method that returns a red fail
    # usage:
    #   print(bcolors.fail("What you did just failed massively. Bummer"))
    #   or:
    #   sys.exit(bcolors.fail("Not a valid date"))
    @staticmethod
    def fail(message):
        return bcolors.FAIL + message + bcolors.ENDC

    # Method that returns a green ok
    # usage:
    #   print(bcolors.ok("What you did just ok-ed massively. Yay!"))
    @staticmethod
    def ok(message):
        return bcolors.OKGREEN + message + bcolors.ENDC

    # Method that returns a blue ok
    # usage:
    #   print(bcolors.okblue("What you did just ok-ed into the blue. Wow!"))
    @staticmethod
    def okblue(message):
        return bcolors.OKBLUE + message + bcolors.ENDC

    # Method that returns a header in some purple-ish color
    # usage:
    #   print(bcolors.header("This is great"))
    @staticmethod
    def header(message):
        return bcolors.HEADER + message + bcolors.ENDC


def parsArgs():
    parser = argparse.ArgumentParser(description="PyTorch LapSRN Test")
    parser.add_argument("--scale", default=4, type=int,
                        help="scale factor, Default: 4")
    parser.add_argument("--isTest", type=bool,
                        default=False, help="Test or not")
    parser.add_argument('--dataset', type=str, default='SOTS',
                        help='Path of the validation dataset')
    parser.add_argument("--checkpoint", default="models/MSBDN-DFF/1/model.pkl",
                        type=str, help="Test on intermediate pkl (default: none)")
    parser.add_argument('--gpu_ids', type=str, default='0',
                        help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--name', type=str, default='MSBDN',
                        help='filename of the training models')
    parser.add_argument("--start", type=int, default=2,
                        help="Activated gate module")
    parser.add_argument("--input", type=str, default="input",
                        help="input image / folder")
    parser.add_argument("--output", type=str,
                        default="output", help="output image / folder")

    return parser.parse_args()


def is_pkl(filename):
    return any(filename.endswith(extension) for extension in [".pkl"])


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def get_folder(path):
    for filename in glob.iglob(path + '**/*.jpg', recursive=True):
        print(filename)
    return filename


def print_all_args(args):
    print(">>>Printing all arguments")
    for arg in vars(args):
        print(bcolors.warning(">>> " + arg), " | ", getattr(args, arg))


def load_folder(path):
    tmp = tqdm(path)
    print("Loading images from {}".format(path))


def create_folder(path):
    os.makedirs(path, exist_ok=True)
    print(">>>Folder created".format(path))


def get_last_path_name(path):
    return path.split("/")[-1]


def model_test(model, drive):
    model = model.to(device)
    criterion = torch.nn.MSELoss(size_average=True)
    criterion = criterion.to(device)
    return criterion


def test(test_gen, model, criterion, SR_dir):
    avg_psnr = 0
    avg_ssim = 0
    med_time = []

    with torch.no_grad():
        for iteration, batch in enumerate(test_gen, 1):
            # print(iteration)
            Blur = batch[0]
            HR = batch[1]
            Blur = Blur.to(device)
            HR = HR.to(device)

            name = batch[2][0][:-4]

            # -------------------------begin to deal with an image's time
            start_time = time.perf_counter()

            sr = model(Blur)

            # modify
            try:
                sr = torch.clamp(sr, min=0, max=1)
            except:
                sr = sr[0]
                sr = torch.clamp(sr, min=0, max=1)
            torch.cuda.synchronize()  # wait for CPU & GPU time syn
            evalation_time = time.perf_counter() - start_time  # ---------finish an image
            med_time.append(evalation_time)

            ssim = pytorch_ssim.ssim(sr, HR)
            # print(ssim)
            avg_ssim += ssim

            mse = criterion(sr, HR)
            psnr = 10 * log10(1 / mse)
            #
            resultSRDeblur = transforms.ToPILImage()(sr.cpu()[0])
            resultSRDeblur.save(
                join(SR_dir, '{0}_{1}.png'.format(name, opt.name)))

            print("Processing {}:  PSNR:{} TIME:{}".format(
                iteration, psnr, evalation_time))
            avg_psnr += psnr

        print("===> Avg. SR SSIM: {:.4f} ".format(avg_ssim / iteration))
        print("Avg. SR PSNR:{:4f} dB".format(avg_psnr / iteration))
        median_time = statistics.median(med_time)
        print(median_time)
        return avg_psnr / iteration


print("--------- Dehazing v1.0 ---------")
args = parsArgs()
print_all_args(args)

input_file_path = get_folder(args.input)
print(input_file_path)


print(bcolors.header("------ start ------"))

print(bcolors.header("------ Loading CUDA ------"))
device = torch.device('cuda:{}'.format(
    args.gpu_ids[0])) if torch.cuda.is_available() else torch.device('cpu')
str_ids = args.gpu_ids.split(',')
torch.cuda.set_device(int(str_ids[0]))
testloader = DataLoader(DataValSet(input_file_path),
                        batch_size=1, shuffle=False, pin_memory=False)
print(bcolors.ok("------ CUDA loaded ------"))


print(bcolors.header("------ creating output ------"))
output_file_path = args.output
create_folder(output_file_path)
print(bcolors.ok("The results of testing images stored in {}.".format(output_file_path)))


print(bcolors.header("------ Loading model ------"))
if is_pkl(args.checkpoint):
    test_pkl = args.checkpoint
    if is_pkl(test_pkl):
        print("Testing model {}----------------------------------".format(args.checkpoint))
        model = torch.load(test_pkl, map_location=lambda storage, loc: storage)
        print(get_n_params(model))
        #model = model.eval()
        criterion = model_test(model , device)
        print("criterion: {}".format(criterion))
        # psnr = test(testloader, model, criterion, output_file_path)
    else:
        print("It's not a pkl file. Please give a correct pkl folder on command line for example --opt.checkpoint /models/1/GFN_epoch_25.pkl)")


# load_folder(path_lists)


# Citation:
#     Gated Fusion Network for Joint Image Deblurring and Super-Resolution
#     The British Machine Vision Conference(BMVC2018 oral)
#     Xinyi Zhang, Hang Dong, Zhe Hu, Wei-Sheng Lai, Fei Wang and Ming-Hsuan Yang
# Contact:
#     cvxinyizhang@gmail.com
# Project Website:
#     http://xinyizhang.tech/bmvc2018
#     https://github.com/jacquelinelala/GFN


# def get_n_params(model):
#     pp = 0
#     for p in list(model.parameters()):
#         nn = 1
#         for s in list(p.size()):
#             nn = nn*s
#         pp += nn
#     return pp


# def is_pkl(filename):
#     return any(filename.endswith(extension) for extension in [".pkl"])


# def test(test_gen, model, criterion, SR_dir):
#     avg_psnr = 0
#     avg_ssim = 0
#     med_time = []

#     with torch.no_grad():
#         for iteration, batch in enumerate(test_gen, 1):
#             # print(iteration)
#             Blur = batch[0]
#             HR = batch[1]
#             Blur = Blur.to(device)
#             HR = HR.to(device)

#             name = batch[2][0][:-4]

#             # -------------------------begin to deal with an image's time
#             start_time = time.perf_counter()

#             sr = model(Blur)

#             # modify
#             try:
#                 sr = torch.clamp(sr, min=0, max=1)
#             except:
#                 sr = sr[0]
#                 sr = torch.clamp(sr, min=0, max=1)
#             torch.cuda.synchronize()  # wait for CPU & GPU time syn
#             evalation_time = time.perf_counter() - start_time  # ---------finish an image
#             med_time.append(evalation_time)

#             ssim = pytorch_ssim.ssim(sr, HR)
#             # print(ssim)
#             avg_ssim += ssim

#             mse = criterion(sr, HR)
#             psnr = 10 * log10(1 / mse)
#             #
#             resultSRDeblur = transforms.ToPILImage()(sr.cpu()[0])
#             resultSRDeblur.save(
#                 join(SR_dir, '{0}_{1}.png'.format(name, opt.name)))

#             print("Processing {}:  PSNR:{} TIME:{}".format(
#                 iteration, psnr, evalation_time))
#             avg_psnr += psnr

#         print("===> Avg. SR SSIM: {:.4f} ".format(avg_ssim / iteration))
#         print("Avg. SR PSNR:{:4f} dB".format(avg_psnr / iteration))
#         median_time = statistics.median(med_time)
#         print(median_time)
#         return avg_psnr / iteration


# def model_test(model):
#     model = model.to(device)
#     criterion = torch.nn.MSELoss(size_average=True)
#     criterion = criterion.to(device)
#     print(opt)
#     psnr = test(testloader, model, criterion, SR_dir)
#     return psnr


# opt = parser.parse_args()
# device = torch.device('cuda:{}'.format(
#     opt.gpu_ids[0])) if torch.cuda.is_available() else torch.device('cpu')
# str_ids = opt.gpu_ids.split(',')
# torch.cuda.set_device(int(str_ids[0]))
# root_val_dir = opt.dataset  # ----------Validation path
# # --------------------------SR results save path
# SR_dir = join(root_val_dir, 'Results')
# isexists = os.path.exists(SR_dir)
# if not isexists:
#     os.makedirs(SR_dir)
# print("The results of testing images sotre in {}.".format(SR_dir))

# testloader = DataLoader(DataValSet(root_val_dir),
#                         batch_size=1, shuffle=False, pin_memory=False)
# print("===> Loading model and criterion")

# if is_pkl(opt.checkpoint):
#     test_pkl = opt.checkpoint
#     if is_pkl(test_pkl):
#         print("Testing model {}----------------------------------".format(opt.checkpoint))
#         model = torch.load(test_pkl, map_location=lambda storage, loc: storage)
#         print(get_n_params(model))
#         #model = model.eval()
#         model_test(model)
#     else:
#         print("It's not a pkl file. Please give a correct pkl folder on command line for example --opt.checkpoint /models/1/GFN_epoch_25.pkl)")
# else:
#     test_list = [x for x in sorted(os.listdir(opt.checkpoint)) if is_pkl(x)]
#     print("Testing on the given 3-step trained model which stores in /models, and ends with pkl.")
#     Results = []
#     Max = {'max_psnr': 0, 'max_epoch': 0}
#     for i in range(len(test_list)):
#         print(
#             "Testing model is {}----------------------------------".format(test_list[i]))
#         print(join(opt.checkpoint, test_list[i]))
#         model = torch.load(
#             join(opt.checkpoint, test_list[i]), map_location=lambda storage, loc: storage)
#         print(get_n_params(model))
#         model = model.eval()
#         psnr = model_test(model)
#         Results.append(
#             {'epoch': "".join(re.findall(r"\d", test_list[i])[:]), 'psnr': psnr})
#         if psnr > Max['max_psnr']:
#             Max['max_psnr'] = psnr
#             Max['max_epoch'] = "".join(re.findall(r"\d", test_list[i])[:])
#     for Result in Results:
#         print(Result)
#     print('Best Results is : ===========================> ')
#     print(Max)


# Citation:
#     Gated Fusion Network for Joint Image Deblurring and Super-Resolution
#     The British Machine Vision Conference(BMVC2018 oral)
#     Xinyi Zhang, Hang Dong, Zhe Hu, Wei-Sheng Lai, Fei Wang and Ming-Hsuan Yang
# Contact:
#     cvxinyizhang@gmail.com
# Project Website:
#     http://xinyizhang.tech/bmvc2018
#     https://github.com/jacquelinelala/GFN
