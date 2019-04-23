import numpy as np
class Neuron(object):
    def __init__(self,inputMatrix,numHidden,outputMatrix):
        self.inputMatrix = inputMatrix
        self.NH = numHidden
        self.outputMatrix = outputMatrix
        self.wOne = self.weightMatrix(self.inputMatrix,self.NH)
        self.wTwo = self.weightMatrix(self.fillerMatrix(),len(self.outputMatrix[0]))
        self.iterations = 100
        self.n = .1
    def fillerMatrix(self):
        column = [[]]
        for i in range(self.NH):
            column[0].append(1)
        return column
    def weightMatrix(self,x,z):
        column = []
        for i in range(len(x[0])):
            row = []
            for j in range(z):
                row.append(np.random.randn())
            column.append(row)
        return column
    def dotProduct(self,x,w,N):
        column = []
        for i in range(N):
            row = []
            for j in range(len(x)):
                t = 0
                for k in range(len(x[0])):
                    t += x[j][k]*w[k][i]
                row.append(t)
            column.append(row)
        return column
    def sigmoid(self,z,der=False):
        if der == True:
            for i in range(len(z)):
                for n in range(len(z[0])):
                    z[i][n] = np.exp(z[i][n])/((1+np.exp(-z[i][n]))**2)
        else:
            for j in range(len(z)):
                for p in range(len(z[0])):
                    z[j][p] = 1/(1+np.exp(-z[j][p]))
        return z
    def dotB(self,x,w):
        column = []
        for i in range(len(self.outputMatrix)):
            row = []
            for j in range(1):
                t = 0
                for k in range(self.NH):
                    t += x[k][i]*w[k][0]
                row.append(t)
            column.append(row)
        return column
    def forwardPropogation(self,x):
        self.a = self.dotProduct(x,self.wOne,self.NH)
        self.aSig = self.sigmoid(self.a)
        self.b = self.dotB(self.aSig,self.wTwo)
        self.bSig = self.sigmoid(self.b)
        self.yHat = self.bSig
        return self.yHat
    def dEdyHat(self,yHat):
        sumderror = 0
        for i in range(len(self.outputMatrix)):
            sumderror += -(self.outputMatrix[i][0]-yHat[i][0])
        return sumderror
    def dyHatdb(self):
        return self.sigmoid(self.b,der=True)
    def dbdw2(self):
        product = []
        for i in range(len(self.a[0])):
            plus = 0
            for j in range(len(self.a)):
                plus += (self.a[i][j])
            product.append(plus)
        return product
    def chainRule(self,yHat):
        total = []
        for j in range(len(self.dyHatdb())):
            total.append(0)
        return total
    def deltaW2(self,yHat):
        for i in range(len(self.wTwo)):
            self.wTwo[i][0] = self.wTwo[i][0]-(self.n*((self.dEdyHat(yHat)*self.dbdw2()[i]*self.dyHatdb()[i][0])))
    def deltaW1(self):
        pass



x = [[5,9],[2,4],[8,3]]
x = x/np.amax(x)
y = [[76],[64],[89]]
y = y/np.amax(y)

NN = Neuron(x,3,y)
yHat = NN.forwardPropogation(x)
yHat
print(NN.wTwo)

NN.dbdw2()
NN.dyHatdb()
NN.deltaW2(yHat)
print(NN.wTwo)
