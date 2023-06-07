import pandas as pd
from data import data
from model import model
from accuracy import accuracy

df_merged = pd.read_csv("--------------------") #dataset path 

def main():
    tax, tex, tay, tey, en, ee = data(df_merged)
    b,r,s,t= model(tax, tex, tay, tey, ee,df_merged)
    l = accuracy(b,r,s,t)
    print("\nAccuracy: ",l)
    
if __name__ == "__main__":
    main()