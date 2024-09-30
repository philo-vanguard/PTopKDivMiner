import pandas as pd
import argparse
import random
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Given a dataset, get the sample one")
    parser.add_argument('-data_name', '--data_name', type=str, default="airports")
    parser.add_argument('-data', '--data', type=str, default="../datasets/airports/airports.csv")
    parser.add_argument('-sample_num', '--sample_num', type=int, default=20000)
    parser.add_argument('-sample_data', '--sample_data', type=str, default="../datasets/airports/airports_sample.csv")

    args = parser.parse_args()
    arg_dict = args.__dict__
    for k, v in sorted(arg_dict.items()):
        print("k:", k, ", v:", v)

    data = pd.read_csv(arg_dict["data"])

    random.seed(1234567)
    all_index = [i for i in range(data.shape[0])]
    sample_size = arg_dict["sample_num"]
    choosen_index = np.array(random.sample(all_index, sample_size))
    print(choosen_index.shape)
    print(choosen_index[:10])

    revised_attrs = None
    if arg_dict["data_name"] == "airports":
        revised_attrs = ["id", "elevation_ft"]
    elif arg_dict["data_name"] == "ncvoter":
        revised_attrs = ["id", "county_id"]
    elif arg_dict["data_name"] == "hospital":
        revised_attrs = ["ZIP_Code", "Phone_Number"]
    elif "tax" in arg_dict["data_name"]:
        revised_attrs = ["areacode", "phone", "zip", "salary", "singleexemp", "marriedexemp", "childexemp"]
    elif "aminer" in arg_dict["data_name"]:
        if "AMiner_Author2Paper" in arg_dict["data"]:
            revised_attrs = ["author2paper_id", "author_id", "paper_id", "author_position"]
        elif "AMiner_Author" in arg_dict["data"]:
            revised_attrs = ["author_id", "published_papers", "citations", "h_index"]
        elif "AMiner_Paper" in arg_dict["data"]:
            revised_attrs = ["paper_id", "year"]
    elif "adults_categorical" in arg_dict["data_name"]:
        revised_attrs = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
    elif "adults" in arg_dict["data_name"]:
        revised_attrs = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
    
    if revised_attrs is not None:
        for attr in revised_attrs:
            data[attr] = data[attr].astype(str).str.replace('\.0$', '').replace("nan", np.nan)

    data.loc[choosen_index].to_csv(arg_dict["sample_data"], index=False)

