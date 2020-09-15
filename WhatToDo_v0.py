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

class dataNormalization:
    ''' Data normalization class'''
    def __init__(self, data):
        self.initialized = False
        if not type(data) is np.ndarray:
            raise ValueError('dataNormalization required numpy array for initialization')
        self.dataStat = stats.describe(data)
        self.initialized = True

    def normalize(self, data):
        return (data - self.dataStat.mean) / np.sqrt(self.dataStat.variance)


class loadData:
    ''' load data functionality '''
    def __init__(self, trainFileName, evalFileName, PATH="C:\\Users\\tzurv\\python\\VScode\\scraper\\"):
        print(f"# Loading train data file {PATH+TrainFileName}")
        self.allTrainingData =  np.genfromtxt(PATH+TrainFileName, delimiter=',') 

        print(f"# Loading eval data file {PATH+EvalFileName}")
        self.allEvalData     =  np.genfromtxt(PATH+EvalFileName, delimiter=',') 
        
    def getRawTrain(self):
        return self.allTrainingData
        
    def getRawEval(self):
        # return without fund name
        return self.allEvalData[:,1:]

    def getEvalFundByLocation(self, indx):
        return self.allEvalData[indx,0]
        
    def seperateOutput(self, data):
        return data[:,0:-1], data[:,-1]


class selectFeatues:
    
    def VIFbasedSelection(analyzedData):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            indxs = [x for x in range(analyzedData.shape[1])]
            #for L in range(3, analyzedData.shape[1]):
            Half = int(analyzedData.shape[1] / 2)
            for L in range(Half+3 , Half-1, -1):
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

    
class trainingClass:
    def __init__(self):
        # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
        self.is_cuda = torch.cuda.is_available()

        # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
        if self.is_cuda:
            self.device = torch.device("cuda")
            print("# GPU is available")
        else:
            self.device = torch.device("cpu")
            print("# GPU not available, CPU used")    


    def prepare(self, xtrain, ytrain):
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
        self.model = TwoLayerNet(self.D_in, self.H, self.D_out)
        self.model = self.model.to(self.device)


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
    
    def my_loss(self, output, target):
        loss = 0.0
        for o, t in zip(output, target):
            if t[0]<=10.0 :
                loss += (o[0]-t[0])**2
                
            else:
                loss += (abs(o[0]-t[0])**1.5)
        #loss = torch.mean((output - target)**2)
        return loss

    def my_loss_v1(self, output, target, midRanveLimit=10):
        loss = 0.0
        for o, t in zip(output, target):
            dif = abs(o[0]-t[0])
            if t[0]<0.0 :
                if o[0]<0.0:
                    loss += (dif**1.5)
                    
                elif o[0]<midRanveLimit:
                    loss += (dif**2)

                else:
                    loss += (dif**3)
                
            elif t[0]<midRanveLimit:
                if o[0]<0.0:
                    loss += (dif**3)
                    
                elif o[0]<midRanveLimit:
                    loss += (dif**2)

                else:
                    loss += (dif**1.5)
                            
            else:
                if o[0]<0.0:
                    loss += (dif**3)
                    
                elif o[0]<midRanveLimit:
                    loss += (dif**1.5)

                else:
                    loss += (dif**1.1)
                
        return loss
    
    def train(self, iters):

        for t in range(iters):
            # Forward pass: Compute predicted y by passing x to the model
            self.y_pred = self.model(self.x)
            #print(self.y_pred[:10], self.y[:10])

            # Compute and print loss
            self.loss = self.criterion(self.y_pred, self.y)
            #self.loss = self.my_loss(self.y_pred, self.y)
            #self.loss = self.my_loss_v1(self.y_pred, self.y)
            
            if t % 500 == 99 :
                print(f"\t {t} {self.loss.item():0.4f}")
                #for p, r in zip(self.y_pred[:10], self.y[:10]):
                #    print(f"\tpredicted {p[0]:.2f} vs out {r[0]:.2f}")

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()
        
        return self.model           


    def evaluate(self, model, evalX, evalY, title="Scatter plot real vs predicted.", midRanveLimit=10):

        print(f"Eval title {title}, size {evalY.shape}, midRanveLimit={midRanveLimit}")
        
        # evaluate
        with torch.no_grad():
            localEvalX = torch.from_numpy(evalX.astype(np.float32))
            localEvalX = localEvalX.to(self.device)
            out = model(localEvalX)

        out = out.to('cpu')
        out = torch.squeeze((out)) 
        npPredicted = out.detach().numpy()

        if False:
            colors = (0,0,0)
            area = np.pi*3
            plt.errorbar(evalY, npPredicted, yerr=evalY-npPredicted, fmt='o')
            plt.errorbar(evalY, evalY, fmt='x', markeredgecolor = 'r')
            plt.title(title)
            plt.xlabel('real')
            plt.ylabel('predicted')
            plt.show()    

        # analyzed results 
        # 3 bands: negative, 0<=x<midRanveLimit, midRanveLimit<=xlabel 
        #midRanveLimit = 20
        I = pd.Index(['Rnegative','RmidRange','RHigh'], name="rows")
        C = pd.Index(['Pnegative','PmidRange','PHigh', 'cases', 'MSError'], name="columns")
        dfConfusion = pd.DataFrame(data=np.zeros(shape=(3,5)), index=I, columns=C)
        for p, r in zip(npPredicted, evalY):
            pRange = 'PHigh'
            if p<=0:
                pRange='Pnegative'
            elif p<midRanveLimit:
                pRange='PmidRange'
           
            if r<0:
                dfConfusion[pRange]['Rnegative'] += 1
                dfConfusion['cases']['Rnegative'] += 1
                dfConfusion['MSError']['Rnegative'] += (p-r)**2
            elif r<midRanveLimit:
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
        print(f"Done. \n")
        
       

class parserClass:
    #https://realpython.com/python-command-line-arguments/
    def __init__(self, args_dict):
        parser = argparse.ArgumentParser(description='Process funds data')
        #parser.add_argument('integers', metavar='N', type=int, nargs='+',
        #           help='an integer for the accumulator')
        parser.add_argument('--task',     '-t', help='define the required task', default='train')
        parser.add_argument('--date',     '-d', help='data files date name', required=True)
        parser.add_argument('--features', '-f', help='selected features, example -f "[1, 2, 3, 4]"', default='all')

        self.args = parser.parse_args(args_dict)



if __name__ == "__main__":

    # Date information 
    dateNow = datetime.datetime.now()
    print(f"running at {dateNow}")

    if len(sys.argv)>1:
        localParser = parserClass(sys.argv[1:])
    else:
        localParser = parserClass(['-h'])
    print(f"Task: {localParser.args.task}")    
    


    # load data files
    dateStamp = localParser.args.date
    TrainFileName = dateStamp + "_Train.csv"
    EvalFileName  = dateStamp + "_Eval.csv"
    allRawData = loadData(TrainFileName, EvalFileName)
    allTrainData = allRawData.getRawTrain()
    allEvalData = allRawData.getRawEval()
    
    # Seperate traing for output
    trainX, trainYpart = allRawData.seperateOutput(allTrainData)
    
    # Normalize data
    dNorm = dataNormalization(trainX)
    normTrainX = dNorm.normalize(trainX)
    
    if localParser.args.task == "select":
        print("Start to select data.")
        selectFeatues.VIFbasedSelection(normTrainX)
        sys.exit()
        
    elif localParser.args.task == "train":
        print("Train model.")
        col_idx = None
        if not localParser.args.features == 'all':
            # expected input -f "[1, 2, 3, 4]"
            col_idx = [int(x) for x in localParser.args.features.strip('][').split(', ')]
            col_idx = np.array(col_idx)
            print(f"# Selected features are {col_idx}")
            normTrainX = normTrainX[:,col_idx]
        
        # create and check if cuda is available
        trainer = trainingClass()
        
        # create model, move data and model to device
        trainer.prepare(normTrainX, trainYpart)
        trainedModel = trainer.train(50000)
        midRanveLimit = 20
        trainer.evaluate(trainedModel, normTrainX, trainYpart, title="Train Data",  midRanveLimit=midRanveLimit)

        # Seperate traing for output
        print(allEvalData.size)
        evalX, evalYpart = allRawData.seperateOutput(allEvalData)
        
        
        normEvalX = dNorm.normalize(evalX)
        if not localParser.args.features == 'all':
            normEvalX = normEvalX[:,col_idx]
        trainer.evaluate(trainedModel, normEvalX, evalYpart, title="Evaluation",  midRanveLimit=midRanveLimit)
       
        sys.exit()
    
    print(f"# Unsupported task {localParser.args.task}")
    parserClass(['-h'])
    
    
