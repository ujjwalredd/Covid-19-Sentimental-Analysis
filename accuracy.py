def accuracy(a,b,c,d):
    max_value = a
    print("\nBest Model is : Bidirectional Encoder Representations from Transformers")
    if b > max_value:
        max_value = b
        print("\nBest Model is : RoBERTa: A Robustly Optimized BERT Pretraining Approach")
    if c > max_value:
        max_value = c
        print("\nBest Model is : Sequential Model")
    if d > max_value:
        max_value = d
        print("\nBest Model is : Topic Modelling")
    return max_value

