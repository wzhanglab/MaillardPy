from maillardpy.properties import get_chemopy_props_from_smilesfile
from maillardpy.model import Model
import pandas as pd
import argparse
import os

# User input
parser = argparse.ArgumentParser()
parser.add_argument("input", help="Enter the path to input file")
parser.add_argument("dtype", help="Enter the molecular format of file (smiles/sdf/mol)")
parser.add_argument("output", help="Enter the path to the folder where output will be saved.")

# Model and model feature paths.
ages_MODEL = 'maillardpy/models/ages_chemopy_rf_boruta.p'
ages_FEATURES = 'maillardpy/models/ages_chemopy_boruta_features.p'
noages_MODEL = 'maillardpy/models/noages_chemopy_rf_boruta.p'
noages_FEATURES = 'maillardpy/models/noages_chemopy_boruta_features.p'

def get_prediction(f, dtype):
    
    
    if dtype == 'smiles':
        data = get_chemopy_props_from_smilesfile(f)
    else:
        raise Exception("{} is not a valid molecular format".format(dtype))
    

    # Convert to dataframe
    data = pd.DataFrame(data)

    # Find null rows
    null_values = data.isnull().sum(axis=1)
    null_rows = data.loc[null_values > 0]

    # Drop null rows
    data = data.loc[null_values == 0]

    # Generate prediction
    model = Model(ages_MODEL, ages_FEATURES,
                  noages_MODEL, noages_FEATURES)
    ages_taste, ages_prob, noages_taste, noages_prob = model.predict(data)

    data['ages_taste'] = ages_taste
    data['ages_prob'] = ages_prob[:, 1]

    data['noages_taste'] = noages_taste
    data['noages_prob'] = noages_prob[:, 1]

    return data, null_rows['name'].tolist()

if __name__ == "__main__":
    args = vars(parser.parse_args())
    data, not_predicted = get_prediction(args["input"], args["dtype"])
    data.to_csv(os.path.join(args["output"], 'output.csv'), encoding='utf-8')
    if len(not_predicted) != 0:
        with open(os.path.join(args["output"], 'not_predicted.txt'), 'w') as f:
            for item in not_predicted:
                f.write("{}\n".format(item))
        


