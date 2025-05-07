#Trainer.py
import logging
import os
import shutil
import sys
import numpy as np
import torch

# Disable TensorFloat-32 (TF32) globally for higher precision in GPU computations
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Disable warnings about future deprecation and other non-essential warnings
#torch.set_warn_always(False)
#import warnings
#warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
#import shutup  # External package to suppress warnings
#shutup.please()

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from time import time
from tqdm import tqdm

import multiprocessing
import glob
import random

import torchvision.transforms as transforms
from PIL import Image
from datetime import datetime


# Constants for model training
DISP_CONTENT_STYLE_NUM = 5  # Maximum number of content/style images to display during summary writing
MIN_gradNorm = 0.1  # Minimum gradient norm threshold for clipping
MAX_gradNorm = 1.0  # Maximum gradient norm threshold for clipping

NUM_SAMPLE_PER_EPOCH = 1000  # Number of samples processed per epoch
RECORD_PCTG = NUM_SAMPLE_PER_EPOCH / 5  # Percentage of samples to record during training
eps = 1e-9  # Small value to avoid division by zero in calculations

INITIAL_TRAIN_EPOCHS = 7  # Epoch threshold for transitioning augmentation methods
START_TRAIN_EPOCHS = 3  # Epoch after which simple augmentations start
N_CRITIC = 5


sys.path.append('./')
from Pipelines.Dataset import CharacterDataset
from Utilities.utils import set_random
from Networks.PlainGenerators.PlainWNetBase import WNetGenerator as WNetGenerator
from LossAccuracyEntropy.Loss import Loss
from Utilities.utils import Logging, PrintInfoLog
from Utilities.utils import SplitName

WARMUP_EPOCS = 1
RAMP_EPOCHS = 5

# WNetDict contains two variations of the WNetGenerator model, selecting based on configuration
# WNetDict = {'general': GeneralizedWNet, 'plain': PlainWnet}

# Define data augmentation modes for various stages of training
DataAugmentationMode = {
    'NoAugmentation': ['START', 'START', 'START'],
    'SimpleAugmentation': ['INITIAL', 'INITIAL', 'INITIAL'],
    'HardAumentation': ['FULL', 'FULL', 'FULL'],
    'SimpleAugmentationSchecule': ['START', 'INITIAL', 'INITIAL'],
    'HardAugmentationSchecule': ['START', 'INITIAL', 'FULL'],
}

class ThresholdScheduler:
    """
    Scheduler for dynamically managing threshold values such as gradient norms
    """
    def __init__(self, initial_threshold, decay_factor, min_threshold=0.0001):
        """
        Args:
            initial_threshold (float): Initial threshold value.
            decay_factor (float): Factor by which the threshold is decayed.
            min_threshold (float): Minimum value the threshold can reach.
        """
        self.threshold = initial_threshold
        self.decay_factor = decay_factor
        self.min_threshold = min_threshold

    def Step(self):
        """Decays the threshold value at each step, ensuring it doesn't fall below the minimum threshold."""
        self.threshold = max(self.min_threshold, self.threshold * self.decay_factor)

    def GetThreshold(self):
        """Returns the current threshold value."""
        return self.threshold


class Trainer(nn.Module):
    """
    Trainer class responsible for managing the entire training process, including data loading,
    model initialization, optimization, and logging.
    """
    def __init__(self, hyperParams=-1, penalties=-1):
        super().__init__()

        # Store hyperparameters and penalties
        self.config = hyperParams
        self.penalties = penalties

        # If not resuming training, clean up the existing log, image, and experiment directories
        if self.config.userInterface.resumeTrain == 0:
            if os.path.exists(self.config.userInterface.logDir):
                shutil.rmtree(self.config.userInterface.logDir)
            if os.path.exists(self.config.userInterface.trainImageDir):
                shutil.rmtree(self.config.userInterface.trainImageDir)
            if os.path.exists(self.config.userInterface.expDir):
                shutil.rmtree(self.config.userInterface.expDir)

        # Create directories for logs, training images, and experiment data if they don't exist
        os.makedirs(self.config.userInterface.logDir, exist_ok=True)
        os.makedirs(self.config.userInterface.trainImageDir, exist_ok=True)
        os.makedirs(self.config.userInterface.expDir, exist_ok=True)
        os.makedirs(self.config.userInterface.expDir+'/Generator', exist_ok=True)
        os.makedirs(self.config.userInterface.expDir+'/Discriminator', exist_ok=True)
        os.makedirs(self.config.userInterface.expDir+'/Frameworks', exist_ok=True)
        

        # Initialize TensorBoard writer for tracking training progress
        self.writer = SummaryWriter(self.config.userInterface.logDir)

        # Configure logging to include timestamps (year, month, day, hour, minute, second)
        current_time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        logging.basicConfig(filename=os.path.join(self.config.userInterface.logDir, current_time + "-Log.txt"),
                            level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')

        # Create console handler to display logs in the terminal
        self.sessionLog = Logging(sys.stdout)
        self.sessionLog.terminator = ''  # Prevent automatic newline in log output
        logging.getLogger().addHandler(self.sessionLog)

        # Data augmentation strategy based on the configuration
        self.augmentationApproach = DataAugmentationMode[self.config.datasetConfig.augmentation]
        self.debug = self.config.debug  # Flag to enable debug mode
        set_random()  # Set random seed for reproducibility

        self.iters = 0  # Initialize iteration counter
        self.startEpoch = 0  # Initialize starting epoch

        # Model initialization: select the appropriate WNetGenerator model based on configuration
        self.generator = WNetGenerator(self.config, self.sessionLog)
        self.generator.train()  # Set the model to training mode
        self.generator.cuda()  # Move the model to GPU
        
        

        # Apply Xavier initialization to layers (Conv2D and Linear)
        for m in self.generator.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)  # Set biases to 0
            elif isinstance(m, (nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)  # Set biases to 0
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)  # Set weights to 1 for BatchNorm layers
                m.bias.data.zero_()  # Set biases to 0 for BatchNorm layers

        # Optimizer selection based on the config (Adam, SGD, or RMSprop)
        if self.config.trainParams.optimizer == 'adam':
            self.optimizerG = torch.optim.Adam(self.generator.parameters(), lr=self.config.trainParams.initLr, betas=(0.0, 0.9),
                                              weight_decay=self.penalties.PenaltyGeneratorWeightRegularizer)
        elif self.config.trainParams.optimizer == 'sgd':
            self.optimizerG = torch.optim.SGD(self.generator.parameters(), lr=self.config.trainParams.initLr, momentum=0.9)
        elif self.config.trainParams.optimizer == 'rms':
            self.optimizerG = torch.optim.RMSprop(self.generator.parameters(), lr=self.config.trainParams.initLr,
                                                 alpha=0.99, eps=1e-08,
                                                 weight_decay=self.penalties.PenaltyGeneratorWeightRegularizer)
            

        # Loss function
        self.sumLoss = Loss(self.config, self.sessionLog, self.penalties, [WARMUP_EPOCS,RAMP_EPOCHS])

        # Set number of workers for loading data based on the debug mode
        if not self.debug:
            workersNum = multiprocessing.cpu_count() // 3  # Use one-third of the available CPU cores
            workersNum = 24
        else:
            workersNum = 1  # No multiprocessing in debug mode
        # Log the number of threads used for reading data
        PrintInfoLog(self.sessionLog, f"Reading Data: {workersNum}/{multiprocessing.cpu_count()} Threads")

        # Initialize training and testing datasets and data loaders
        self.trainset = CharacterDataset(self.config, sessionLog=self.sessionLog)
        self.trainLoader = DataLoader(self.trainset, batch_size=self.config.trainParams.batchSize,
                                      num_workers=workersNum, pin_memory=True, 
                                      shuffle=True, drop_last=False, 
                                      persistent_workers=True)

        self.testSet = CharacterDataset(self.config, sessionLog=logging.info, is_train=False)
        self.testLoader = DataLoader(self.testSet, batch_size=self.config.trainParams.batchSize,
                                     num_workers=workersNum, pin_memory=True, 
                                     shuffle=False, drop_last=False, 
                                     persistent_workers=True)

        # Learning rate scheduler with exponential decay
        lrGamma = np.power(0.01, 1.0 / (self.config.trainParams.epochs - 1))
        self.lrScheculerG = torch.optim.lr_scheduler.ExponentialLR(gamma=lrGamma, optimizer=self.optimizerG)
        # self.lrScheculerD = torch.optim.lr_scheduler.ExponentialLR(gamma=lrGamma, optimizer=self.optimizerD)

        # Gradient norm scheduler
        gradNormGamma = np.power(0.75, 1.0 / (self.config.trainParams.epochs - 1))
        self.gradGNormScheduler = ThresholdScheduler(MIN_gradNorm, gradNormGamma)

        # Resume training from the latest checkpoint if required
        if self.config.userInterface.resumeTrain == 1:
            PrintInfoLog(self.sessionLog, f'Load model from {self.config.userInterface.expDir}')
            
            # for generator 
            listFiles = glob.glob(self.config.userInterface.expDir+'/Generator' + '/*.pth')
            latestFile = max(listFiles, key=os.path.getctime)  # Get the latest checkpoint file
            ckptG = torch.load(latestFile)  # Load the checkpoint
            self.generator.load_state_dict(ckptG['stateDict'])  # Load the model weights
            self.optimizerG.load_state_dict(ckptG['optimizer'])  # Load the optimizer state
            
            # for discriminator 
            listFiles = glob.glob(self.config.userInterface.expDir+'/Discriminator' + '/*.pth')
            latestFile = max(listFiles, key=os.path.getctime)  # Get the latest checkpoint file
            ckptD = torch.load(latestFile)  # Load the checkpoint
            
            # for framework
            listFiles = glob.glob(self.config.userInterface.expDir+'/Frameworks' + '/*.pth')
            latestFile = max(listFiles, key=os.path.getctime)  # Get the latest checkpoint file
            ckptFramework = torch.load(latestFile)  # Load the checkpoint
            self.startEpoch = ckptFramework['epoch']  # Get the starting epoch from the checkpoint
            
            

            # Reset learning rate after loading
            for param_group in self.optimizerG.param_groups:
                param_group['lr'] = self.config.trainParams.initLr
                
            for param_group in self.optimizerD.param_groups:
                param_group['lr'] = self.config.trainParams.initLr

            # Apply the learning rate and gradient schedulers up to the start epoch
            for _ in range(self.startEpoch):
                self.gradGNormScheduler.Step()  # Adjust gradient norm threshold
                self.lrScheculerG.step()  # Adjust learning rate
            self.startEpoch = self.startEpoch

        # Initialize a dictionary to record gradient values
        self.gradG = {}

        class _gradient():
            def __init__(self):
                self.value = 0.0
                self.count = 0

        # Populate the gradient dictionary with parameters from the model
        for idx, (name, param) in enumerate(self.generator.named_parameters()):
            subName, layerName = name.split('.')[0], name.split('.')[1]
            if subName not in self.gradG:
                self.gradG.update({subName: {}})
            if layerName not in self.gradG[subName]:
                self.gradG[subName].update({layerName: _gradient()})
            self.gradG[subName][layerName].count = self.gradG[subName][layerName].count + 1

        

        # Log that the trainer is ready
        PrintInfoLog(self.sessionLog, 'Trainer prepared.')

    def ClearGradRecords(self):
        """Reset gradient records to 0 for all layers."""
        for idx1, (subName, subDict) in enumerate(self.gradG.items()):
            for idx2, (key, value) in enumerate(self.gradG[subName].items()):
                self.gradG[subName][key].value = 0.0

    def SummaryWriting(self, evalContents, evalStyles, evalGTs, evalFakes, epoch, step, 
                       lossG, lossDict, mark='NA', writeImageToDisk=False):
        """
        Write summaries to TensorBoard including images and scalar losses.

        Args:
            evalContents (Tensor): Content inputs.
            evalStyles (Tensor): Style inputs.
            evalGTs (Tensor): Ground truth images.
            evalFakes (Tensor): Generated images.
            step (int): Current step or epoch.
            lossG (float): Generator loss.
            lossDict (dict): Dictionary of individual loss components.
            mark (str): Mark for identifying if it's 'Train' or 'Test' phase.
        """
        dispContentNum = min(DISP_CONTENT_STYLE_NUM, evalContents.shape[1])
        dispStyleNum = min(DISP_CONTENT_STYLE_NUM, evalStyles.shape[1])
        selectedContentIdx = random.sample(range(evalContents.shape[1]), dispContentNum)
        selectedStyleIdx = random.sample(range(evalStyles.shape[1]), dispStyleNum)
        outGrids = []

        # Create image grids for TensorBoard display
        B = evalContents.shape[0]
        for bid in range(B):
            fake = evalFakes[bid]
            gt = evalGTs[bid]
            difference = torch.abs(fake - gt)
            contents = evalContents[bid][selectedContentIdx].unsqueeze(1)
            styles = evalStyles[bid][selectedStyleIdx].unsqueeze(1)
            outList = []
            for x in contents:
                outList.append(x)
            outList.append(gt)
            outList.append(difference)
            outList.append(fake)
            for x in styles:
                outList.append(x)

            outGrid = make_grid([x for x in outList], nrow=dispContentNum + dispStyleNum + 3, normalize=True, scale_each=True)
            outGrids.append(outGrid)
        output = make_grid(outGrids, nrow=1, normalize=True, scale_each=True)
        tensor2Img = transforms.ToPILImage()
        outImg = tensor2Img(output)
        

        # Write images to TensorBoard and disk
        if mark == 'Train':
            self.writer.add_image("TrainImage", output, dataformats='CHW', global_step=step)
        elif mark == 'Test':
            self.writer.add_image("TestImage", output, dataformats='CHW', global_step=step)
        
        # Save Grid Image to Disk 
        if writeImageToDisk: 
            outImg.save(os.path.join(self.config.userInterface.trainImageDir, "%sEpoch%d.png" % (mark, epoch)))


        # Write scalar losses to TensorBoard
        self.writer.add_scalar('01-LossGenerator/SumLossG-' + mark, lossG, global_step=step)
        self.writer.add_scalar('01-LossReconstruction/L1-' + mark, lossDict['lossL1'], global_step=step)
        self.writer.add_scalar('01-LossGenerator/ConstContentReal-' + mark, lossDict['lossConstContentReal'], global_step=step)
        self.writer.add_scalar('01-LossGenerator/ConstStyleReal-' + mark, lossDict['lossConstStyleReal'], global_step=step)
        self.writer.add_scalar('01-LossGenerator/ConstContentFake-' + mark, lossDict['lossConstContentFake'], global_step=step)
        self.writer.add_scalar('01-LossGenerator/ConstStyleFake-' + mark, lossDict['lossConstStyleFake'], global_step=step)
        self.writer.add_scalar('01-LossGenerator/CategoryRealContent-' + mark, lossDict['lossCategoryContentReal'], global_step=step)
        self.writer.add_scalar('01-LossGenerator/CategoryFakeContent-' + mark, lossDict['lossCategoryContentFake'], global_step=step)
        self.writer.add_scalar('01-LossGenerator/CategoryRealStyle-' + mark, lossDict['lossCategoryStyleReal'], global_step=step)
        self.writer.add_scalar('01-LossGenerator/CategoryFakeStyle-' + mark, lossDict['lossCategoryStyleFake'], global_step=step)
        self.writer.add_scalar('01-LossReconstruction/DeepPerceptualContentSum-' + mark, lossDict['deepPerceptualContent'], global_step=step)
        self.writer.add_scalar('01-LossReconstruction/DeepPerceptualStyleSum-' + mark, lossDict['deepPerceptualStyle'], global_step=step)
        
        # 记录 KL Loss（如果存在）
        if 'kl_loss' in lossDict:
            for ii in range(len(lossDict['kl_loss'])):
                self.writer.add_scalar('02-LossKL/KL%d'% (ii+1) + '-' + mark , lossDict['kl_loss'][ii], global_step=step)
            
        if mark == 'Train':
            # Clear previous gradient records
            self.ClearGradRecords()
            for idx, (name, param) in enumerate(self.generator.named_parameters()):
                if param.grad is not None:
                    subName, layerName = name.split('.')[0], name.split('.')[1]
                    self.gradG[subName][layerName].value = self.gradG[subName][layerName].value + torch.norm(param.grad)

            # Log gradient norms to TensorBoard
            self.writer.add_scalar('00-GradientCheck/00-MinGradThreshold', self.gradGNormScheduler.GetThreshold(), global_step=step)
            self.writer.add_scalar('00-GradientCheck/00-LearningRate', self.lrScheculerG.get_lr()[0], global_step=step)
            for idx1, (subName, subDict) in enumerate(self.gradG.items()):
                for idx2, (layerName, value) in enumerate(self.gradG[subName].items()):
                    currentName = subName + '-' + layerName
                    self.writer.add_scalar('00-GradientCheck/' + currentName, 
                                           self.gradG[subName][layerName].value / self.gradG[subName][layerName].count * self.lrScheculerG.get_lr()[0], 
                                           global_step=step)
            

        # Log deep perceptual loss if extractors are used
        if 'extractorContent' in self.config:
            for idx, thisContentExtractor in enumerate(self.config.extractorContent):
                thisContentExtractorName = thisContentExtractor.name
                self.writer.add_scalar('011-LossDeepPerceptual-ContentMSE-' + mark + '/' + thisContentExtractorName, lossDict['deepPerceptualContentList'][idx], global_step=step)

        if 'extractorStyle' in self.config:
            for idx, thisStyleExtractor in enumerate(self.config.extractorStyle):
                thisStyleExtractorName = thisStyleExtractor.name
                self.writer.add_scalar('013-LossDeepPerceptual-StyleMSE-' + mark + '/' + thisStyleExtractorName, lossDict['deepPerceptualStyleList'][idx], global_step=step)

        self.trainStart = time()  # Log the time when training starts
        self.writer.flush()
        
    def TrainOneEpoch(self, epoch):
        # reset stylelist
        self.trainset.ResetStyleList(epoch+1,"Train")
        
        
        # torch.autograd.set_detect_anomaly(True)
        """Train the model for a single epoch."""
        self.generator.train()  # Set the model to training mode
        # self.discriminator.train()
        time1 = time()  # Track start time for the epoch
        thisRoundStartItr = 0  # Used to track when to write summaries
        
                
        
        trainProgress = tqdm(enumerate(self.trainLoader), total=len(self.trainLoader),
                             desc="Training @ Epoch %d" % (epoch+1))  # Progress bar for the training epoch
        
        for idx, (contents, styles, gt, onehotContent, onehotStyle) in trainProgress:
            # Move data to the GPU and require gradients
            contents, styles, gt, onehotContent, onehotStyle = contents.cuda().requires_grad_(), \
                styles.cuda().requires_grad_(), gt.cuda().requires_grad_(), \
                onehotContent.cuda().requires_grad_(), onehotStyle.cuda().requires_grad_()
            reshapedStyle = styles.reshape(-1, 1, 64, 64)

            # Forward pass: generate content and style features, categories, and final output (fake images)
            encodedContentFeatures, encodedStyleFeatures, encodedContentCategory, encodedStyleCategory, generated, vae = \
                self.generator(contents, reshapedStyle, gt)

            # Prepare inputs for the loss function
            lossInputs = {'InputContents': contents, 
                          'InputStyles': reshapedStyle, 
                          'GT': gt, 'fake':generated,
                          'encodedContents':[encodedContentFeatures,encodedContentCategory],
                          'encodedStyles':[encodedStyleFeatures,encodedStyleCategory],
                          'oneHotLabels':[onehotContent, onehotStyle]}

            # Compute the total generator loss and detailed loss breakdown
            sumLossG, Loss_dict = self.sumLoss(epoch, lossInputs, 
                                                         SplitName(self.config.generator.mixer)[1:-1],
                                                        #  critic=self.discriminator,
                                                         mode='Train')
            

            # Update the progress bar with the current loss
            dispID = self.config.expID.replace('Exp','')
            dispID = dispID.replace('Encoder','E')
            dispID = dispID.replace('Mixer','M')
            dispID = dispID.replace('Decoder','D')
            trainProgress.set_description(dispID + " Training @ Epoch: %d/%d, LossL1: %.3f" %
                                          (epoch+1, self.config.trainParams.epochs, Loss_dict['lossL1']), refresh=True)

            
            self.optimizerG.zero_grad()
            sumLossG.backward()  # Compute gradients
            
        
            # Gradient norm clipping and adjustment (if enabled)
            if self.config.trainParams.gradientNorm:
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=MAX_gradNorm)  # Clip gradients
                for name, param in self.generator.named_parameters():
                    if param.grad is not None:
                        gradNorm = torch.norm(param.grad)
                        if gradNorm < self.gradGNormScheduler.GetThreshold():
                            param.grad = param.grad + eps  # Prevent zero gradients by adding epsilon
                            gradNorm = torch.norm(param.grad)  # Recompute gradient norm
                            scaleFactor = self.gradGNormScheduler.GetThreshold() / (gradNorm + eps)  # Adjust scale factor
                            param.grad = param.grad*scaleFactor  # Scale up the gradient
            
            self.optimizerG.step()  # Update model parameters
            self.iters = self.iters+ 1  # Update iteration count

            # Log and write summaries at regular intervals
            if (idx * float(NUM_SAMPLE_PER_EPOCH) / len(self.trainLoader) - thisRoundStartItr > RECORD_PCTG//10 and epoch < START_TRAIN_EPOCHS) or \
                (idx * float(NUM_SAMPLE_PER_EPOCH) / len(self.trainLoader) - thisRoundStartItr > RECORD_PCTG//5 and epoch < INITIAL_TRAIN_EPOCHS) or \
                (idx * float(NUM_SAMPLE_PER_EPOCH) / len(self.trainLoader) - thisRoundStartItr > RECORD_PCTG) or \
                idx == 0 or idx == len(self.trainLoader) - 1:
                thisRoundStartItr = idx * float(NUM_SAMPLE_PER_EPOCH) / len(self.trainLoader)
                self.SummaryWriting(evalContents=contents, evalStyles=styles, evalGTs=gt, evalFakes=generated, epoch=epoch+1,
                                    step=epoch * NUM_SAMPLE_PER_EPOCH + int(idx / (len(self.trainLoader)) * NUM_SAMPLE_PER_EPOCH)+1,
                                    lossG=sumLossG, lossDict=Loss_dict, mark='Train', writeImageToDisk=idx==len(self.trainLoader)-2)

        time2 = time()  # Track end time for the epoch
        PrintInfoLog(self.sessionLog, 'Training completed @ Epoch: %d/%d training time: %f mins, L1Loss: %.3f' % 
                     (epoch+1, self.config.trainParams.epochs, (time2-time1)/60, Loss_dict['lossL1']))  # Log epoch time and loss

    def TestOneEpoch(self, epoch):
        
        # reset stylelist
        self.testSet.ResetStyleList(epoch+1, "Test")
        
        """Evaluate the model on the test dataset for a single epoch."""
        is_train = False  # Disable training mode for evaluation
        self.generator.eval()  # Set the model to evaluation mode
        # self.discriminator.eval()
        time1 = time()  # Track start time
        with torch.no_grad():  # Disable gradient calculations for efficiency
            thisRoundStartItr = 0
            testProgress = tqdm(enumerate(self.testLoader), total=len(self.testLoader), desc="Testing @ Epoch %d" % (epoch+1))
            for idx, (contents, styles, gt, onehotContent, onehotStyle) in testProgress:
                # Move data to the GPU
                contents, styles, gt, onehotContent, onehotStyle = contents.cuda(), \
                    styles.cuda(), gt.cuda(), onehotContent.cuda(), onehotStyle.cuda()

                # Reshape style input
                # reshaped_styles = styles.reshape(self.config.trainParams.batchSize * self.config.datasetConfig.inputStyleNum, 1, 64, 64)
                reshaped_styles = styles.reshape(-1, 1, 64, 64)

                # Forward pass
                encodedContentFeatures, encodedStyleFeatures, encodedContentCategory, encodedStyleCategory, generated, vae = \
                    self.generator(contents, reshaped_styles, gt, is_train=is_train)


                # Prepare inputs for the loss function
                lossInputs = {'InputContents': contents, 
                            'InputStyles': reshaped_styles, 
                            'GT': gt, 'fake':generated,
                            'encodedContents':[encodedContentFeatures,encodedContentCategory],
                            'encodedStyles':[encodedStyleFeatures,encodedStyleCategory],
                            'oneHotLabels':[onehotContent, onehotStyle]}
                
                
                sumLossG, Loss_dict = self.sumLoss(epoch, lossInputs,
                                                             SplitName(self.config.generator.mixer)[1:-1],
                                                            #  critic=self.discriminator, 
                                                             mode='Test')

                
                # Update progress bar with current loss
                dispID = self.config.expID.replace('Exp','')
                dispID = dispID.replace('Encoder','E')
                dispID = dispID.replace('Mixer','M')
                dispID = dispID.replace('Decoder','D')
                testProgress.set_description(dispID + " Testing @ Epoch: %d/%d, LossL1: %.3f" %
                                             (epoch+1, self.config.trainParams.epochs, Loss_dict['lossL1']), refresh=True)

                # Write summaries at regular intervals
                if (idx * float(NUM_SAMPLE_PER_EPOCH) / len(self.testLoader) - thisRoundStartItr > RECORD_PCTG and epoch < START_TRAIN_EPOCHS) or \
                    (idx * float(NUM_SAMPLE_PER_EPOCH) / len(self.testLoader) - thisRoundStartItr > RECORD_PCTG*2) or \
                    idx == 0 or idx == len(self.testLoader) - 1:
                    thisRoundStartItr = idx * float(NUM_SAMPLE_PER_EPOCH) / len(self.testLoader)
                    self.SummaryWriting(evalContents=contents, evalStyles=styles, evalGTs=gt, evalFakes=generated,epoch=epoch+1,
                                        step=epoch * NUM_SAMPLE_PER_EPOCH + int(idx / len(self.testLoader) * NUM_SAMPLE_PER_EPOCH)+1,
                                        lossG=sumLossG, lossDict=Loss_dict, mark='Test', writeImageToDisk=idx==len(self.testLoader)-2)

        time2 = time()  # Track end time for the epoch
        PrintInfoLog(self.sessionLog, 'Testing completed @ Epoch: %d/%d testing time: %f mins, L1Loss: %.3f' % 
                     (epoch+1, self.config.trainParams.epochs, (time2-time1)/60, Loss_dict['lossL1']))  # Log epoch time and loss

    def Pipelines(self):
        """Main training and evaluation loop."""
        train_start = time()  # Start time of the entire training process
        training_epoch_list = range(self.startEpoch, self.config.trainParams.epochs, 1)  # List of epochs to train

        if (not self.config.userInterface.skipTest) and self.startEpoch != 0:
            self.TestOneEpoch(self.startEpoch-1)  # Test at the start if not skipping

        for epoch in training_epoch_list:
            # Reset the data augmentation if resuming training from an early epoch
            if self.startEpoch < START_TRAIN_EPOCHS:
                self.trainset.ResetTrainAugment(info=self.augmentationApproach[0])
            elif self.startEpoch < INITIAL_TRAIN_EPOCHS:
                self.trainset.ResetTrainAugment(info=self.augmentationApproach[1])
            else:
                self.trainset.ResetTrainAugment(info=self.augmentationApproach[2])

            # Train and test model at each epoch
            self.TrainOneEpoch(epoch)
            if not self.config.userInterface.skipTest and\
                (epoch < START_TRAIN_EPOCHS \
                    or (epoch < INITIAL_TRAIN_EPOCHS and epoch % 3 == 0) \
                    or (epoch % 5 == 0) \
                    or epoch == self.config.trainParams.epochs-1):
                self.TestOneEpoch(epoch)

            # Save model checkpoint at the end of the epoch
            stateG = {'stateDict': self.generator.state_dict(),'optimizer': self.optimizerG.state_dict()}
            torch.save(stateG, self.config.userInterface.expDir+'/Generator' + '/CkptEpoch%d.pth' % (epoch+1))
            stateFramework = {'epoch': epoch+1}
            torch.save(stateFramework, self.config.userInterface.expDir+'/Frameworks' + '/CkptEpoch%d.pth' % (epoch+1))
            
            
            logging.info(f'Trained model has been saved at Epoch {epoch+1}.')

            # Step the learning rate scheduler and gradient norm scheduler
            self.lrScheculerG.step()
            self.gradGNormScheduler.Step()

        # After training, log the total time taken and close the TensorBoard writer
        train_end = time()  # Record the end time of training
        training_time = (train_end - train_start) / 3600  # Convert total training time to hours
        self.writer.close()  # Close the TensorBoard writer
        logging.info('Training finished, tensorboardX writer closed')
        logging.info('Training total time: %f hours.' % training_time)
