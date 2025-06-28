import json
import pandas as pd
import os

class Parser:
    def __new__(cls, jsonPath):
        jsonData = None
        try:
            with open(jsonPath, 'r') as f:
                data = json.load(f)
                jsonData = data['messages']
        except Exception as e:
            print("Error in opening JSON: " + str(e))

        dfs = {} 
        if jsonData:
            for code in jsonData.keys():
                try:
                    # Ensure the data is a list of dictionaries before creating DataFrame
                    # todo- make this modular, fix it so that it can 
                    # clean a variety of dataframes
                    data_list = jsonData[code]
                    for field in data_list.keys():
                        if type(data_list[field]) == list:
                            jsonData[code][field] = {str(ix) : jsonData[code][field][ix] for ix in range(len(jsonData[code][field]))}
                    """
                    if code == 'FILE':
                        jsonData[code]['FileName'] = {str(ix) : jsonData[code]['FileName'][ix] for ix in range(len(jsonData[code]['FileName']))}
                        jsonData[code]['Data'] = {str(ix) : jsonData[code]['Data'][ix] for ix in range(len(jsonData[code]['Data']))}
                    if code == 'PARM':
                        jsonData[code]['Name'] = {str(ix) : jsonData[code]['Name'][ix] for ix in range(len(jsonData[code]['Name']))}
                    """
                    dfs[code] = pd.DataFrame(data_list)
                except Exception as e:
                    print("Error in creating dataframe with code " + str(code) + ": " + str(e))
        return dfs