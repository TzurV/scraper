# -*- coding: utf-8 -*

# https://docs.python.org/3.3/library/argparse.html
import argparse
# https://pytorch.org/get-started/locally/#mac-anaconda
# conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
import torch
from torch import nn
import numpy as np
import datetime
from scipy import stats
import sys
import itertools
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
import matplotlib.pyplot as plt
import os

class dataNormalization:
    ''' Data normalization class'''
    def __init__(self, data, Path=None):
        self.initialized = False
        if Path is not None:
            self.loadStat(Path)
        else:
            if not type(data) is np.ndarray:
                raise ValueError('dataNormalization required numpy array for initialization')
            self.dataStat = stats.describe(data)
            self.mean = self.dataStat.mean
            self.variance = self.dataStat.variance
        self.initialized = True

    def normalize(self, data):
        return (data - self.mean) / np.sqrt(self.variance)
    
    def saveStat(self, filepath):
        '''
        documentaion https://numpy.org/devdocs/reference/generated/numpy.save.html
        save mean and variance
        '''
        fileName = os.path.join(filepath, 'stat.npy')
        with open(fileName,'wb') as f:
            np.save(f, self.mean)
            np.save(f, self.variance)
 
    def loadStat(self, filepath):
        fileName = os.path.join(filepath, 'stat.npy')
        with open(fileName,'rb') as f:
            self.mean = np.load(f)            
            self.variance = np.load(f)            

class loadData:
    ''' load data functionality '''
    def __init__(self, trainFileName, evalFileName,\
                 PredictFile=None, PATH="C:\\Users\\tzurv\\python\\VScode\\scraper\\"):
        
        print(f"# Loading train data file {PATH+TrainFileName}")
        self.allTrainingData =  np.genfromtxt(PATH+TrainFileName, delimiter=',') 

        print(f"# Loading eval data file {PATH+EvalFileName}")
        self.allEvalData     =  np.genfromtxt(PATH+EvalFileName, delimiter=',') # , dtype=None
        
        if PredictFile is not None:
            print(f"# Loading predict data file {PATH+PredictFile}")
            self.allPredictData  =  np.genfromtxt(PATH+PredictFile, delimiter=',', dtype=str) 
            
            
        
    def getRawTrain(self):
        return self.allTrainingData
        
    def getRawEval(self):
        # return without fund name
        return self.allEvalData[:,1:]

    def getEvalFundByLocation(self, indx):
        return self.allEvalData[indx,0]
        
    def seperateOutput(self, data):
        return data[:,0:-1], data[:,-1]
    
    def getDataForPrediction(self):
        '''
        Returns
        -------
            funds names (class 'numpy.ndarray'>), raw data (class 'numpy.ndarray'>)
        '''
        return self.allPredictData[:,0], self.allPredictData[:, 1:-1].astype(float)


class selectFeatues:
    
    def VIFbasedSelection(analyzedData):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            indxs = [x for x in range(analyzedData.shape[1])]
            #for L in range(3, analyzedData.shape[1]):
            Half = int(analyzedData.shape[1] / 2)
            foundSubset = False
            for L in range(Half+2 , Half-1, -1):
                print(f" --------------- {L} ------------")
                for c in itertools.combinations(indxs, L):
                    col_idx = np.asarray(c)
                    partData = analyzedData[:,col_idx]
                    pdTrain = pd.DataFrame(data=partData, columns=[str(x) for x in range(partData.shape[1])])
                    X = pdTrain
                    vif = pd.DataFrame()
                    vif["features"] = col_idx
                    vif["vif_Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]    
                    if sum(vif["vif_Factor"]<10) == L:
                        print(vif)   
                        foundSubset = True
                        
                if foundSubset:
                    break
    

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out, dropout=0.2):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.relu1    = torch.nn.ReLU(inplace=True)
        #The dropout module nn.Dropout conveniently handles this and shuts dropout off as soon as your model enters evaluation mode
        self.dropout1 = torch.nn.Dropout(p=dropout)
        self.linear2 = torch.nn.Linear(H, H)
        self.relu2    = torch.nn.ReLU(inplace=True)
        self.linear3 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h_relu1 = self.linear1(x)
        h_relu1 = self.relu1(h_relu1)
        h_relu1 = self.dropout1(h_relu1)
        h_relu2 = self.linear2(h_relu1)
        h_relu2 = self.relu2(h_relu2)
        y_pred  = self.linear3(h_relu2)
        return y_pred

def setDevice():
    # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
        print("# GPU is available")
    else:
        device = torch.device("cpu")
        print("# GPU not available, CPU used")   

    return device
    
    

class trainingClass:
    def __init__(self, dNorm):
        
        # store a copy of data normalization object
        self.dNorm = dNorm
        
        # # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
        # self.is_cuda = torch.cuda.is_available()

        # # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
        # if self.is_cuda:
        #     self.device = torch.device("cuda")
        #     print("# GPU is available")
        # else:
        #     self.device = torch.device("cpu")
        #     print("# GPU not available, CPU used")  
        
        # select device to run (GPU or CPU)
        self.device = setDevice()
            
        # initialize
        self.hparams = {}

    @staticmethod
    def createModel(hparams, device):
        model = TwoLayerNet(hparams['D_in'], 
                                 hparams['H'], 
                                 hparams['D_out'], 
                                 dropout=hparams['dropout'])
        return model.to(device)

    def prepare(self, xtrain, ytrain, dropout=0.2):
        # change numpy representation to float and conver to pytorch tensor
        self.x = torch.from_numpy(xtrain.astype(np.float32))
        self.y = torch.from_numpy(ytrain.astype(np.float32))
        
        # add a Dimension
        self.y = self.y.unsqueeze(1)

        print(self.x.shape, self.y.shape)

        # N is batch size; D_in is input dimension;
        # H is hidden dimension; D_out is output dimension.
        self.N, self.D_in, self.H, self.D_out = self.x.shape[0], self.x.shape[1], 25, 1
        print(f"N={self.N}, D_in={self.D_in}, H={self.H}, D_out={self.D_out}")   

        # move to device 
        self.x = self.x.to(self.device)
        self.y = self.y.to(self.device) 

        print(f"# input dim {self.x.shape}")

        # Construct our model by instantiating the class defined above
        self.hparams['D_in'] = self.D_in
        self.hparams['H'] = self.H
        self.hparams['D_out'] = self.D_out
        self.hparams['dropout'] = dropout
        if False:
            self.model = TwoLayerNet(self.hparams['D_in'], 
                                     self.hparams['H'], 
                                     self.hparams['D_out'], 
                                     dropout=self.hparams['dropout'])
            self.model = self.model.to(self.device)
        else:
            self.model = trainingClass.createModel(self.hparams, self.device)

        #print the model
        print(self.model)
        modelParameters = sum([param.nelement() for param in self.model.parameters()])
        print('Num Model Parameters ', modelParameters )

        # Construct our loss function and an Optimizer. The call to model.parameters()
        # in the SGD constructor will contain the learnable parameters of the two
        # nn.Linear modules which are members of the model.

        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0000001, momentum=0.90, weight_decay=0.0001)
        #optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        
    def model_hparameters():
        return self.hparams
    
    def my_loss(self, output, target):
        loss = 0.0
        for o, t in zip(output, target):
            if t[0]<=10.0 :
                loss += (o[0]-t[0])**2
                
            else:
                loss += (abs(o[0]-t[0])**1.5)
        #loss = torch.mean((output - target)**2)
        return loss

    def my_loss_v1(self, output, target, midRangeLimit=10):
        loss = 0.0
        for o, t in zip(output, target):
            dif = abs(o[0]-t[0])
            if t[0]<0.0 :
                if o[0]<0.0:
                    loss += (dif**1.5)
                    
                elif o[0]<midRangeLimit:
                    loss += (dif**2)

                else:
                    loss += (dif**3)
                
            elif t[0]<midRangeLimit:
                if o[0]<0.0:
                    loss += (dif**3)
                    
                elif o[0]<midRangeLimit:
                    loss += (dif**2)

                else:
                    loss += (dif**1.5)
                            
            else:
                if o[0]<0.0:
                    loss += (dif**3)
                    
                elif o[0]<midRangeLimit:
                    loss += (dif**1.5)

                else:
                    loss += (dif**1.1)
                
        return loss
    
    def train(self, epochs, outPath,  evalData=None):
        
        reportEvery = int(epochs/50)
        

        lowestLoss = 1e+308
        lowestLossOutputDir = None
        lastSavedLoss = lowestLoss
        for t in range(epochs):
            # save eval results for refernce (for verification at load time)
            storeEvalResulrs = (int(epochs/2) == t)
            
            # Forward pass: Compute predicted y by passing x to the model
            self.y_pred = self.model(self.x)

            # Compute and print loss
            self.loss = self.criterion(self.y_pred, self.y)
            #self.loss = self.my_loss(self.y_pred, self.y)
            #self.loss = self.my_loss_v1(self.y_pred, self.y)
            if t == 0:
                lowestLoss = self.loss
                prevLoss   = self.loss
                lastSavedLoss = self.loss + 100.0
            elif (lowestLoss < self.loss and lowestLoss==prevLoss \
                and t>epochs/2) or storeEvalResulrs:
                # models are saved once we done ove  half of the epochs to reduce number of save models
                # current model loss is higer the the previous one and 
                # previous one was the best one so far
                lossGain = 1-(lastSavedLoss-prevLoss)/lastSavedLoss
                if outPath is not None:
                    # https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
                    lowestLossOutputDir = outPath + "\\" + str(t-1)
                    os.makedirs(lowestLossOutputDir)
                    
                    modelFileName = lowestLossOutputDir + "\\model.pt"
                    
                    print(f"\t# Saving epoch {t-1} loss {prevLoss} lossGain {lossGain}")
                    EPOCH = t-1
                    LOSS = prevLoss
                    torch.save({
                                'epoch': EPOCH,
                                'model_state_dict': self.model.state_dict(),
                                #'optimizer_state_dict': optimizer.state_dict(),
                                'loss': LOSS,
                                **self.hparams,
                                }, modelFileName)
                    # save data normalization statistics to a file
                    self.dNorm.saveStat(lowestLossOutputDir)
                    
                    # evaluate model with eval data 
                    if evalData is not None and storeEvalResulrs:
                        curStdout = sys.stdout
                        evalOutputFileName = os.path.join(lowestLossOutputDir, "eval.txt")
                        
                        # redirect stdoutput to a file
                        sys.stdout = open(evalOutputFileName,"w")
                        print (f"stdout is redirected to a file {t} {evalOutputFileName}")
 
                        # Evaluate eval data
                        self.evaluate(self.model, evalDic['EvalX'], evalDic['evalY'], 
                                         title=evalDic['title'],  
                                         midRangeLimit=evalDic['midRangeLimit'])
                        
                        # stop rediredtion of print to stdout to a file
                        sys.stdout.close()    
                        sys.stdout = curStdout
                    
            prevLoss   = self.loss
            if lowestLoss>prevLoss:
                lowestLoss = prevLoss
            
            if t / reportEvery == int(t/reportEvery) :
                print(f"\t {t} {self.loss.item():0.4f}")
                #lastSavedLoss *= 2.0
                #for p, r in zip(self.y_pred[:10], self.y[:10]):
                #    print(f"\tpredicted {p[0]:.2f} vs out {r[0]:.2f}")

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()
        
        return self.model, lowestLossOutputDir           


    def evaluate(self, model, evalX, evalY, title="Scatter plot real vs predicted.", midRangeLimit=10,\
                 confusionMatrix=True):

        print(f"Eval title {title}, midRangeLimit={midRangeLimit}")
        
        # evaluate
        with torch.no_grad():
            localEvalX = torch.from_numpy(evalX.astype(np.float32))
            localEvalX = localEvalX.to(self.device)
            out = model(localEvalX)

        out = out.to('cpu')
        out = torch.squeeze((out)) 
        npPredicted = out.detach().numpy()

        # if False:
        #     colors = (0,0,0)
        #     area = np.pi*3
        #     plt.errorbar(evalY, npPredicted, yerr=evalY-npPredicted, fmt='o')
        #     plt.errorbar(evalY, evalY, fmt='x', markeredgecolor = 'r')
        #     plt.title(title)
        #     plt.xlabel('real')
        #     plt.ylabel('predicted')
        #     plt.show()    

        if confusionMatrix:
            # analyzed results 
            # 3 bands: negative, 0<=x<midRangeLimit, midRangeLimit<=xlabel 
            #midRangeLimit = 20
            I = pd.Index(['Rnegative','RmidRange','RHigh'], name="rows")
            C = pd.Index(['Pnegative','PmidRange','PHigh', 'cases', 'MSError'], name="columns")
            dfConfusion = pd.DataFrame(data=np.zeros(shape=(3,5)), index=I, columns=C)
            for p, r in zip(npPredicted, evalY):
                pRange = 'PHigh'
                if p<=0:
                    pRange='Pnegative'
                elif p<midRangeLimit:
                    pRange='PmidRange'
               
                if r<0:
                    dfConfusion[pRange]['Rnegative'] += 1
                    dfConfusion['cases']['Rnegative'] += 1
                    dfConfusion['MSError']['Rnegative'] += (p-r)**2
                elif r<midRangeLimit:
                    dfConfusion[pRange]['RmidRange'] += 1
                    dfConfusion['cases']['RmidRange'] += 1
                    dfConfusion['MSError']['RmidRange'] += (p-r)**2
                else:
                    dfConfusion[pRange]['RHigh'] += 1
                    dfConfusion['cases']['RHigh'] += 1
                    dfConfusion['MSError']['RHigh'] += (p-r)**2
                    
            # Average
            for a in ['Rnegative', 'RmidRange', 'RHigh']:
                if dfConfusion['cases'][a]>0:
                    dfConfusion['MSError'][a] /= dfConfusion['cases'][a]
            
                    
            # print confusion matrix
            print(dfConfusion)
            print("Done. \n")
        
        return npPredicted
        
       

class parserClass:
    #https://realpython.com/python-command-line-arguments/
    def __init__(self, args_dict):
        parser = argparse.ArgumentParser(description='Process funds data')
        #parser.add_argument('integers', metavar='N', type=int, nargs='+',
        #           help='an integer for the accumulator')
        parser.add_argument('--task',     '-t', help='define the required task', default='train')
        parser.add_argument('--date',     '-d', help='data files date name', required=True)
        parser.add_argument('--features', '-f', help='selected features, example -f "[1, 2, 3, 4]"', default='all')
        parser.add_argument('--epochs', help='number of epochs [50000]', type=int, default=50000)
        parser.add_argument('--dropout', help='Network dropouts', type=float, default=0.2)
        parser.add_argument('--predict', '-p', help='predict filename', default=None)
        parser.add_argument('--modelsPath', '-m', help='Models path', default=None)
        parser.add_argument('--epochNumber', '-e', help='Model epoch directory', default=None)
        
        self.args = parser.parse_args(args_dict)



if __name__ == "__main__":
    # set pytorch random function seed
    torch.manual_seed(0)

    # Date information 
    dateNow = datetime.datetime.now()
    print(f"running at {dateNow}")
    
    # use as timestamp
    current_time = dateNow.strftime("%y%m%d_%H%M")
    
    # parse command line
    if len(sys.argv)>1:
        localParser = parserClass(sys.argv[1:])
    else:
        localParser = parserClass(['-h'])
    print(f"Task: {localParser.args.task}")    

    # load data files
    dateStamp = localParser.args.date
    TrainFileName = dateStamp + "_Train.csv"
    EvalFileName  = dateStamp + "_Eval.csv"
    PredictFileName = None
    if localParser.args.predict is not None:
        PredictFileName = localParser.args.predict+'.csv'
        
    allRawData = loadData(TrainFileName, EvalFileName, PredictFileName)
    allTrainData = allRawData.getRawTrain()
    allEvalData = allRawData.getRawEval()
        
    # Seperate traing for output
    trainX, trainYpart = allRawData.seperateOutput(allTrainData)
    
    # Normalize training data
    if localParser.args.task == "select" or \
        localParser.args.task == "train":
        dNorm = dataNormalization(trainX)
        normTrainX = dNorm.normalize(trainX)
    
    if localParser.args.task == "select":
        print(f"Start to select data. {normTrainX.shape}")
        selectFeatues.VIFbasedSelection(normTrainX)
        sys.exit()
        
    elif localParser.args.task == "train":
        '''
        example:
            python scraper-1\WhatToDo_v0.py  -d 20200926 --predict 20200926_Predict --epochs 50000
        '''

        print("Train model.")
        
        # const
        midRangeLimit = 20

        col_idx = None
        if not localParser.args.features == 'all':
            # expected input -f "[1, 2, 3, 4]"
            col_idx = [int(x) for x in localParser.args.features.strip('][').split(', ')]
            col_idx = np.array(col_idx)
            print(f"# Selected features are {col_idx}")
            normTrainX = normTrainX[:,col_idx]
        
        # create and check if cuda is available
        trainer = trainingClass(dNorm)
        
        # create model, move data and model to device
        trainer.prepare(normTrainX, trainYpart, localParser.args.dropout)
        outPath = "C:\\Users\\tzurv\\python\\VScode\\scraper\\Models"
        outPath +=  "\\" + current_time
        
        # process eval data
        # Seperate traing for output
        print(allEvalData.size)
        evalX, evalYpart = allRawData.seperateOutput(allEvalData)
        
        # normalize eval data
        normEvalX = dNorm.normalize(evalX)
        if not localParser.args.features == 'all':
            normEvalX = normEvalX[:,col_idx]

        # Train model
        evalDic = dict()
        evalDic['EvalX'] = normEvalX
        evalDic['evalY'] = evalYpart
        evalDic['title'] = 'Eval Reference point'
        evalDic['midRangeLimit'] = midRangeLimit        
        trainedModel, bestSavedModelsPath = trainer.train(localParser.args.epochs,
                                                          outPath=outPath,
                                                          evalData=evalDic)
        
        # Evaluate training data
        trainer.evaluate(trainedModel, normTrainX, trainYpart, title="Train Data", midRangeLimit=midRangeLimit)

        # Evaluate eval data
        trainer.evaluate(trainedModel, normEvalX, evalYpart, title="Evaluation",  midRangeLimit=midRangeLimit)
       
        # Now predict
        if PredictFileName is not None:
            fundsNames, predictData = allRawData.getDataForPrediction()
            normPredictData = dNorm.normalize(predictData)
            
            # get relevant data column
            if not localParser.args.features == 'all':
                normPredictData = normPredictData[:,col_idx]

            predictedPerformance = trainer.evaluate(trainedModel, normPredictData, None, title="Predict",\
                                   midRangeLimit=midRangeLimit, confusionMatrix=False )
            for fund, predict in zip(fundsNames, predictedPerformance):
                nextMonth = (pow(1+predict/100, 1/12)-1)*100
                print(f"{nextMonth:6.2f} fund {fund}  ")
            

        sys.exit()

    elif localParser.args.task == "evaluate":
        '''
        example:
            python scraper-1\WhatToDo_v0.py  -t evaluate -d 20200926 --predict 20200926_Predict --modelsPath C:\\Users\\tzurv\\python\\VScode\\scraper\\Models\\201003_1845 -e 26766
        '''
        print("Evaluate model.")
        
        # mandatory command line parameters
        if PredictFileName is None:
            print("\t --predict command line option is mandatory with evaluate option.")
            print("Aborting.")
            sys.exit(1)           
            
        if localParser.args.modelsPath is None:
            print("\t --modelsPath command line option is mandatory with evaluate option.")
            print("Aborting.")
            sys.exit(1)           

        
        #print(localParser.args.modelsPath, localParser.args.epoch)    
        # test loading
        modelFileName = os.path.join(localParser.args.modelsPath, localParser.args.epochNumber, "model.pt")
        print(f"# Loading {modelFileName}")
        checkpoint = torch.load(modelFileName)
        #print(checkpoint.keys())

        # load model        
        device = setDevice()
        model1 = trainingClass.createModel(checkpoint, device)
        model1.load_state_dict(checkpoint['model_state_dict'])
        model1.eval()
        
        # load noramalozation data
        dNorm = dataNormalization(None, os.path.join(localParser.args.modelsPath, localParser.args.epochNumber))
        
        fundsNames, predictData = allRawData.getDataForPrediction()
        normPredictData = dNorm.normalize(predictData)
            
        # get relevant data column
        #if not localParser.args.features == 'all':
        #    normPredictData = normPredictData[:,col_idx]

        trainer = trainingClass(dNorm)

        midRangeLimit = 20
        predictedPerformance = trainer.evaluate(model1, normPredictData, None, title="Predict",\
                               midRangeLimit=midRangeLimit, confusionMatrix=False )
        for fund, predict in zip(fundsNames, predictedPerformance):
            nextMonth = (pow(1+predict/100, 1/12)-1)*100
            print(f"{nextMonth:6.2f} fund {fund}  ")
   
        
        

        sys.exit()

    
    print(f"# Unsupported task {localParser.args.task}")
    parserClass(['-h'])
    
    
