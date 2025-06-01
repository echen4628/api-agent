import json
from utils.constants import (TEXT_EMBEDDING_3_LARGE,
                             INPUTS_DESC_VECTOR_STORE_PATH,
                             OUTPUT_DESC_VECTOR_STORE_PATH,
                             FUNC_DESC_VECTOR_STORE_PATH,
                             NAME_TO_FUNCTION_JSON_PATH)
from utils.FunctionTools import FunctionDatabase
from utils.FunctionWrapper import FunctionWrapper

# open_ai_json = {
#     "name": "Search_Car_Location",
#     "description": "Find car rental location by searching for their name, address, city, state, country, etc.. EndPoint: /api/v1/cars/searchDestination",
#     "parameters": {
#         "type": "object",
#         "properties": {
#             "query": {
#                 "type": "string",
#                 "description": "Names of locations, cities, districts, places, countries, counties etc."
#             }
#         },
#         "required": [
#             "query"
#         ]
#     }
# }
# sample_output = {

#     "data": [
#         {
#             "city": "San Diego",
#             "coordinates": {
#                     "longitude": -117.215935,
#                     "latitude": 32.873055
#             },
#             "country": "United States",
#             "name": "San Diego Marriott La Jolla"
#         }
#     ]

# # }

# Search_Car_Location = FunctionWrapper()
# Search_Car_Location.parse(open_ai_json)
# Search_Car_Location.parse_output(sample_output)

functionDatabase = FunctionDatabase(TEXT_EMBEDDING_3_LARGE, INPUTS_DESC_VECTOR_STORE_PATH,
                                    OUTPUT_DESC_VECTOR_STORE_PATH, FUNC_DESC_VECTOR_STORE_PATH,
                                    NAME_TO_FUNCTION_JSON_PATH)
# functionDatabase.add_function(Search_Car_Location)

# create a vectorstore


def load_json(dir_path):
    if dir_path.endswith('.json'):
        return json.load(open(dir_path, 'r', encoding="utf8"))
    elif dir_path.endswith('.jsonl'):
        return [json.loads(line) for line in open(dir_path, 'r', encoding="utf8")]


data = load_json("../ComplexFuncBench/data/ComplexFuncBench.jsonl")
current_functions = {}
for current_data in data[0:100]:
    conversation_name = current_data["id"]
    conversations = current_data["conversations"]
    functions = current_data["functions"]
    for function in functions:
        current_function = FunctionWrapper()
        current_function.parse(function)
        print(current_function.name)

        current_functions[current_function.name] = current_function
    for i, conversation in enumerate(conversations):
        if conversation['role'] == 'observation':
            assert conversations[i-1]['role'] == 'assistant'
            function_call = conversations[i-1]['function_call']

            num_of_calls = len(function_call)
            for call_num in range(len(function_call)):
                current_function = current_functions[function_call[call_num]['name']]
                if current_function.output is None:
                    current_function.parse_output(
                        {"data": conversation['content'][call_num]['data']})

for key, value in current_functions.items():
    print(f"saving {key}")
    functionDatabase.add_function(value)

# functionDatabase.find_dependency([current_functions['Search_Car_Rentals']])
# response = {'tool_call_id': '${1}', 'role': 'tool', 'name': 'Search_Car_Location', 'content': '{"status": true, "message": "Success", "data": [{"city": "San Diego", "coordinates": {"longitude": -117.215935, "latitude": 32.873055}, "country": "United States", "name": "San Diego Marriott La Jolla"}]}'}