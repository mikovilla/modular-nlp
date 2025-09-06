import pandas as pd

def jsonToCSV(infile, outfile):
    df = pd.read_json(infile)
    df.to_csv(outfile, index=False)
