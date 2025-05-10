from utils.FunctionWrapper import FunctionWrapper

open_ai_json = {
      "name": "Search_Car_Location",
      "description": "Find car rental location by searching for their name, address, city, state, country, etc.. EndPoint: /api/v1/cars/searchDestination",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {
            "type": "string",
            "description": "Names of locations, cities, districts, places, countries, counties etc."
          }
        },
        "required": [
          "query"
        ]
      }
    }
sample_output = {

        "data": [
        {
            "city": "San Diego",
            "coordinates": {
            "longitude": -117.215935,
            "latitude": 32.873055
            },
            "country": "United States",
            "name": "San Diego Marriott La Jolla"
        }
        ]

}

Search_Car_Location = FunctionWrapper()
Search_Car_Location.parse(open_ai_json)
Search_Car_Location.parse_output(sample_output)

from utils.FunctionTools import FunctionDatabase
from utils.constants import (text_embedding_3_large,
                            inputs_desc_vector_store_path,
                            output_desc_vector_store_path,
                            func_desc_vector_store_path,
                            name_to_function_json_path)
functionDatabase = FunctionDatabase(text_embedding_3_large, inputs_desc_vector_store_path,
                                    output_desc_vector_store_path, func_desc_vector_store_path,
                                    name_to_function_json_path)
# functionDatabase.add_function(Search_Car_Location)

# create a vectorstore
import json
def load_json(dir_path):
    if dir_path.endswith('.json'):
        return json.load(open(dir_path, 'r', encoding="utf8"))
    elif dir_path.endswith('.jsonl'):
        return [json.loads(line) for line in open(dir_path, 'r', encoding="utf8")]

data = load_json("../ComplexFuncBench/data/ComplexFuncBench.jsonl")
current_functions = {}
for current_data in data[0:2]:
    conversation_name = current_data["id"]
    conversations = current_data["conversations"]
    functions = current_data["functions"]
    for function in functions:
        current_function = FunctionWrapper()
        current_function.parse(function)
        current_functions[current_function.name] = current_function
    for i, conversation in enumerate(conversations):
        if conversation['role'] == 'observation':
            assert conversations[i-1]['role'] == 'assistant'
            function_call = conversations[i-1]['function_call']

            num_of_calls = len(function_call)
            for call_num in range(len(function_call)):
                current_function = current_functions[function_call[call_num]['name']]
                if current_function.output is None:
                    current_function.parse_output(conversation['content'][call_num]['data'])
        
for key, value in current_functions.items():
    print(f"saving {key}")
    functionDatabase.add_function(value)

functionDatabase.find_dependency([current_functions['Search_Car_Rentals']])