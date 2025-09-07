import pandas as pd

def jsonToCSV(infile, outfile):
    df = pd.read_json(infile)
    df.to_csv(outfile, index=False)

def to_list_str(col):
    out = []
    for x in col:
        if x is None:
            out.append("")
        elif hasattr(x, "as_py"):   # pyarrow scalar -> Python
            out.append(str(x.as_py()))
        else:
            out.append(str(x))
    return out
