from bert import bert
from robert import robert
from sequential import sequential
from topic import topic

def model(tax, tex, tay, tey, ee,df):
    be = bert(tax, tex, tay, tey, ee)
    ro = robert(tax, tex, tay, tey, ee)
    sq = sequential(df)
    tp = topic(df)
    
    return(be,ro,sq,tp)