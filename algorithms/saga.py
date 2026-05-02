import argparse
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import os
os.makedirs("plots", exist_ok=True)
#Noah Jacobson SGD and SAGA
#I added some extra comments
#I had to change it, I realized the outer loop wasn't implemented properly for SARAH
#This is a basic program, it still needs some tuning
#run first:
'''
conda create -n vr_optims python=3.9 pytorch torchvision cpuonly -c pytorch -c conda-forge
conda activate vr_optims
pip install matplotlib tqdm
pip install coloredlogs
pip install scikit-learn
'''
# example  python {{CODENAMEHERE}}.py --method sarah --lr 0.15 --batch-size 512 --epochs 10
#unified optimizer base class
class OptimizerBase:
    # initialized variables
    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr
        self.grads = None
# clears stored and pytorch gradients
    def zero_grad(self):
        self.grads = None
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
#copies and saves gradients
    def store_grads(self):
        self.grads = [p.grad.clone() for p in self.params]
#the vanilla SGD code
class SGD(OptimizerBase):
    def step(self):
        with torch.no_grad():
            for parameter, gradient in zip(self.params, self.grads):
                parameter -=self.lr*gradient
class SARAH(OptimizerBase):
    def get_outer_loop(self):
        #full gradient computed by loss backward
        self.v_prev =[p.grad.clone() for p in self.params]
        self.grads_prev =[p.grad.clone() for p in self.params]
        #one full gradient step, the formula p-=self.lr *v0 is w1=w0-eta*full_gradient
        with torch.no_grad():
            for p, v0 in zip(self.params, self.v_prev):
                p-=self.lr *v0


    #calls the operator constructor
    def __init__(self, params, lr):
        super().__init__(params, lr)
        self.v_prev = None
        self.grads_prev = None
    def step(self):
        with torch.no_grad():
            if self.v_prev is None:
                v = self.grads
                #v=variance reduced gradient, in the first iteration it is simply the regular gradient
            else:
                v = [g-gp+vp for g, gp, vp in zip(self.grads, self.grads_prev, self.v_prev)]
                #v = current gradient - previous gradient+previous variance reduced gradient




            for parameter, v_i in zip(self.params, v):
                parameter-= self.lr * v_i#update parameters with variance reduced gradient
                #v_i is variance reduced gradient for current iteration
            #set previous values for next iteration
            self.v_prev = v
            self.grads_prev = self.grads

class SVRG(OptimizerBase):
    def __init__(self, params, lr):
        super().__init__(params, lr)
        self.full_grads = None
    def get_outer_loop(self, full_grads):
        self.full_grads = full_grads
    def step(self):
        with torch.no_grad():
            if self.full_grads is None:
                v = self.grads
            else:
                v = [g-gp+fg for g, gp, fg in zip(self.grads, self.grads_prev, self.full_grads)]
            for parameter, v_i in zip(self.params, v):
                parameter-= self.lr * v_i
            self.grads_prev = self.grads


#training loop
def train(model, optimizer, loader, epochs, method,dataset):
    history = []#loss per iteration




    for epoch in range(epochs):
        #needed to stabilize training
        #optimizer.v_prev =None
        #optimizer.grads_prev=None
        if (method == "sarah"):
            #additional code needed for outer loop
            optimizer.zero_grad()
            loss =full_loss(model, dataset,lossCriterion=nn.BCEWithLogitsLoss())
            loss.backward()
            optimizer.get_outer_loop()
        for X, y in loader:
            #X and y are current data batch, converted to float tensors
            X, y = X.float(), y.float()#.view(-1, 1) not needed
            #zero grad means that gradients are cleared from previous iteration
            optimizer.zero_grad()
            logits = model(X)
            loss = nn.BCEWithLogitsLoss()(logits, y)
            loss.backward()
            optimizer.store_grads()
            optimizer.step()
            history.append(loss.item())#iteration-level logging




        print(f"Epoch {epoch+1}: loss={loss.item():.4f}")


        #additional statistics
        #eval and no_grad needed for evaluation
        model.eval()
        with torch.no_grad():
            XFull, yFull = dataset[:]
            rawScores = model(XFull) #raw logits
            normalizedProbs = torch.sigmoid(rawScores).cpu().numpy().flatten()
            predictions = (normalizedProbs >= 0.5).astype(int)
            truths = yFull.cpu().numpy().astype(int)
        #get various stats and print them
        accuracy = accuracy_score(truths, predictions)
        precision = precision_score(truths, predictions)
        recall =recall_score(truths, predictions)
        f1 =f1_score(truths, predictions)
        auc =roc_auc_score(truths, normalizedProbs)
        print(f"Epoch {epoch+1} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC: {auc:.4f}")




    return history


def full_loss(model, dataset, lossCriterion):
    XFull, yFull = dataset[:]#works for Tensor dataset
    logits =model(XFull)
    loss= lossCriterion(logits, yFull)
    return loss






#plotting
def plot_loss(history, method):
    plt.figure(figsize=(6,4))
    plt.plot(history, label=method.upper(), linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve ({method.upper()})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/plot_{method}_loss.png")
    print(f"Saved plot to plots/{method}_loss.png")
#main experiment
def main(args):
    #Dataset, changeable (this one now is fairly easy)
    X, y = make_classification(
        n_samples=20000,
        n_features=50,
        n_informative=20,
        n_redundant=5,
        n_repeated=0,
        class_sep=1.5,
        flip_y=0.025,
        random_state=0
    )




    X = StandardScaler().fit_transform(X)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)#unsqueeze makes it (N,1)
    dataset = TensorDataset(X, y)


    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    #model
    model = nn.Linear(X.shape[1], 1)
    # Optimizer selection
    if args.method == "sgd":
        optimizer = SGD(model.parameters(), lr=args.lr)
    elif args.method == "sarah":
        optimizer = SARAH(model.parameters(), lr=args.lr)
    else:#add svrg and saga here later
        raise ValueError("method must be sgd or sarah")




    #Train
    history = train(model, optimizer, loader, args.epochs,method=args.method,dataset=dataset)




    #save results
    with open(f"{args.method}_results.json", "w") as f:
        json.dump({"loss": history}, f, indent=4)




    #plot curve
    plot_loss(history, args.method)




    #Comparison plot, add svrg and saga here later
    methods = ["sgd", "sarah"]
    plt.figure(figsize=(6,4))




    for m in methods:
        try:
            with open(f"{m}_results.json") as f:
                data = json.load(f)
            plt.plot(data["loss"], label=m.upper(), linewidth=2)
        except:
            pass




    plt.xlabel("iteration")
    plt.ylabel("Loss")
    plt.title("SGD vs SARAH")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/plot_comparison.png")
#run in terminal, more functions can be added here for more complex datasets and models
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    main(args)





