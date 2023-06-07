from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



def data(df_merged):
    
    l=['Negative','Positive']
    encoder = LabelEncoder() #label encoding
    df_merged['encoded_sentiment'] = encoder.fit_transform(df_merged['----']) #Sentiment values coumn name change
    encoded_labels = encoder.fit_transform(l)
    

    xtrain, xval, ytrain, yval = train_test_split(df_merged['--------'], df_merged['-----------'], test_size = 0.3, random_state=10) #train_test_split [ Change headline_column ans sentiment respectively.]
    
    return(xtrain,xval,ytrain,yval,encoded_labels,encoder)