from pydantic import BaseModel, create_model, Field
from pydantic.fields import FieldInfo
from typing import List, Annotated, Dict, Union
from typing_extensions import TypedDict
from copy import deepcopy
from utils.constants import type_mapping

def parse_output_to_type_dict(output, name) -> Dict:
    if isinstance(output, list):
        type_dict = {}
        for inner in output:
            temp_dict = parse_output_to_type_dict(inner, name)
            if isinstance(temp_dict[name], dict):
                temp_dict = temp_dict[name]
                for key, value in temp_dict.items():
                    if key not in type_dict:
                        type_dict[key] = value
                    elif type_dict[key] == int and value == float:
                        type_dict[key] = value
            else:
                value = temp_dict[name]
                if value == int:
                    value = float
                if type_dict == {}:
                    type_dict = value
                # elif type_dict == int and value == float:
                #     type_dict = value
            # type_dict |= parse_output_to_type_dict(inner, "")
        return {name+'@d' : type_dict}
    elif not isinstance(output, dict):
        if type(output) == int:
            return {name: float}
        return {name: type(output)}
    else:
        type_dict = {}
        for key, value in output.items():
            type_dict |= parse_output_to_type_dict(value, str(key))
        return {name: type_dict}

def strip_phrase_from_end(s: str, phrase: str):
    if s.endswith(phrase):
        return s[:len(s)-len(phrase)]
    else:
        return s

def parse_type_dict_to_basemodel(output, name):
    # if isinstance(output, list):
    #     # Assume homogeneous list
    #     inner = parse_output_to_basemodel(output[0], "")
    #     return List[inner]
    if not isinstance(output, dict):
        return output
    else:
        fields = {}
        for key, value in output.items():
            field_type = parse_type_dict_to_basemodel(value, strip_phrase_from_end(key, "@d"))
            if "@d" in key:
                fields[strip_phrase_from_end(key, "@d")] = Annotated[List[field_type], Field(default=None)]
                # fields[key.strip("@d")] = Annotated[List[field_type], Field(default=None)]
                # Field()
                # (List[field_type])
            else:
                fields[key] = Annotated[field_type, Field(default=None)]
                # (field_type, ...)
        return create_model(name, **fields)

def parse_output_to_basemodel(output, name):
    # import pdb; pdb.set_trace()
    type_dict = parse_output_to_type_dict(output, name)
    return parse_type_dict_to_basemodel(type_dict[name], name)
    
# def parse_output_to_basemodel(output, name):
#     import pdb; pdb.set_trace()
#     if isinstance(output, list):
#         # Assume homogeneous list
#         inner = parse_output_to_basemodel(output[0], name + "Item")
#         return List[inner]
#     elif not isinstance(output, dict):
#         return type(output)
#     else:
#         fields = {}
#         for key, value in output.items():
#             field_type = parse_output_to_basemodel(value, name + str(key).capitalize())
#             fields[key] = (field_type, ...)
#         return create_model(name, **fields)

def parse_inputs_to_basemodel(open_ai_json_parameter, name):
    fields = {}
    for key, value in open_ai_json_parameter["properties"].items():
        fields[key] = Annotated[type_mapping[value['type']], Field(description=value.get("description", None))]
    return create_model(name, **fields)
            
# def get_leaves_of_basemodel(obj, dict, name) -> Dict:
#     if type(obj) == FieldInfo and obj.annotation in [str, int, float, bool]:
#         return {name: obj.description}
#     elif type(obj) == FieldInfo:
#         return get_leaves_of_basemodel(obj.annotation, name)
#     else:
#         leaf_descriptions = {}
#         for model_field in obj.model_fields:
#             new_name = f"{name}.{model_field}"
#             leaf_descriptions = leaf_descriptions | get_leaves_of_basemodel(obj.model_fields[model_field], new_name)
#     return leaf_descriptions
#     dict[name] = obj.description
#     raise NotImplementedError()


def get_leaves_of_basemodel(obj, name="", prefix_description="") -> Dict[str, str]:
    leaf_descriptions = {}

    if isinstance(obj, FieldInfo):
        # List
        if hasattr(obj.annotation, '__origin__') and issubclass(obj.annotation.__origin__, List):
            annotation = obj.annotation.__args__[0]
            description = obj.description
            
            if annotation in [str, int, float, bool]:
                if prefix_description: 
                    leaf_descriptions[name] = prefix_description + " -> " + (description or name)
                else:
                    leaf_descriptions[name] = description or name
            elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
                leaf_descriptions |= get_leaves_of_basemodel(annotation, name, prefix_description+description if description else prefix_description)
        
        # Primitive field
        elif obj.annotation in [str, int, float, bool, object]:
            if prefix_description: 
                leaf_descriptions[name] = prefix_description + " -> " + (obj.description or name)
            else:
                leaf_descriptions[name] = obj.description or name

        # Nested model
        elif isinstance(obj.annotation, type) and issubclass(obj.annotation, BaseModel):
            leaf_descriptions |= get_leaves_of_basemodel(obj.annotation, name, prefix_description+obj.description if obj.description else prefix_description)
    elif isinstance(obj, type) and issubclass(obj, BaseModel):
        for field_name, field_info in obj.model_fields.items():
            new_name = f"{name}.{field_name}" if name else field_name
            leaf_descriptions |= get_leaves_of_basemodel(field_info, new_name, prefix_description)

    elif hasattr(obj, '__origin__') and issubclass(obj.__origin__, List):
        leaf_descriptions |= get_leaves_of_basemodel(obj.__args__[0], name, prefix_description)
    
    else:
        raise

    return leaf_descriptions

# def extract_from_basemodel(obj, leaf_path):
#     current_obj = obj
#     fields = leaf_path.split(".")
#     for i in range(len(fields)):
#         field = fields[i]
#         if isinstance(current_obj, list):
#             current_obj = [extract_from_basemodel(ele, ".".join(fields[i:])) for ele in current_obj]
#             return_obj = [x[0] for x in current_obj]
#             signatures = [x[1] for x in current_obj]
#             list_signature = signatures[0].split(".")
#             list_signature[0] = list_signature[0] + "@d"
#             return return_obj, ".".join(fields[:i] + list_signature)
#         else:
#             current_obj = getattr(current_obj, field)
#     return current_obj, leaf_path
def extract_from_basemodel_helper(obj, leaf_path):
    current_obj = obj
    fields = leaf_path.split(".")
    try:
        for i in range(len(fields)):
            field = fields[i]
            if isinstance(current_obj, list):
                current_obj = [extract_from_basemodel(ele, ".".join(fields[i:])) for ele in current_obj]
                return_obj = [x[0] for x in current_obj]
                signatures = [x[1] for x in current_obj]
                list_signature = signatures[0].split(".")
                list_signature[0] = list_signature[0] + "@d"
                return return_obj, ".".join(fields[:i] + list_signature)
            else:
                current_obj = getattr(current_obj, field)              
        return current_obj, leaf_path
    except:
        raise 
import difflib

def extract_from_basemodel(obj, leaf_path):
    current_obj = obj
    fields = leaf_path.split(".")
    try:
        for i in range(len(fields)):
            field = fields[i]
            if isinstance(current_obj, list):
                current_obj = [extract_from_basemodel_helper(ele, ".".join(fields[i:])) for ele in current_obj]
                return_obj = [x[0] for x in current_obj]
                signatures = [x[1] for x in current_obj]
                list_signature = signatures[0].split(".")
                list_signature[0] = list_signature[0] + "@d"
                return return_obj, ".".join(fields[:i] + list_signature)
            else:
                current_obj = getattr(current_obj, field)
    except:
        valid_leaves = get_leaves_of_basemodel(obj.__class__)
        valid_leaves.keys()
        suggestions = difflib.get_close_matches(leaf_path, valid_leaves, n=10, cutoff=0.3)
        suggestion_msg = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise AttributeError(f"Path '{leaf_path}' does not exist.{suggestion_msg}")
    return current_obj, leaf_path

def extract_from_dict_helper(obj, leaf_path):
    current_obj = obj
    fields = leaf_path.split(".")
    try:
        for i in range(len(fields)):
            field = fields[i]
            if isinstance(current_obj, list):
                current_obj = [extract_from_dict(ele, ".".join(fields[i:])) for ele in current_obj]
                return_obj = [x[0] for x in current_obj]
                signatures = [x[1] for x in current_obj]
                list_signature = signatures[0].split(".")
                list_signature[0] = list_signature[0] + "@d"
                return return_obj, ".".join(fields[:i] + list_signature)
            else:
                current_obj = current_obj[field]
        return current_obj, leaf_path
    except:
        raise

def get_all_keys(d, prefix=""):
    keys = set()
    if isinstance(d, list):
        for ele in d:
            keys |= get_all_keys(ele, prefix)
    elif not isinstance(d, dict):
        keys.add(prefix)
    else:
        for key in d.keys():
            current_name = prefix+"."+str(key) if prefix else str(key)
            keys |= get_all_keys(d[key], prefix=current_name)
    return keys


def extract_from_dict(obj, leaf_path):
    current_obj = obj
    fields = leaf_path.split(".")
    try:
        for i in range(len(fields)):
            field = fields[i]
            if isinstance(current_obj, list):
                current_obj = [extract_from_dict_helper(ele, ".".join(fields[i:])) for ele in current_obj]
                return_obj = [x[0] for x in current_obj]
                signatures = [x[1] for x in current_obj]
                list_signature = signatures[0].split(".")
                list_signature[0] = list_signature[0] + "@d"
                return return_obj, ".".join(fields[:i] + list_signature)
            else:
                current_obj = current_obj[field]
    except Exception:
        # Gather all valid leaf paths
        import pdb; pdb.set_trace()
       

        valid_leaves = list(get_all_keys(obj))
        suggestions = difflib.get_close_matches(leaf_path, valid_leaves, n=10, cutoff=0.3)
        suggestion_msg = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise KeyError(f"Path '{leaf_path}' does not exist.{suggestion_msg}")
    
    return current_obj, leaf_path

class TestChild(BaseModel):
    child_field: str = Field(default="", description="this is the child field")
  
class TestParent(BaseModel):
    my_child: TestChild = Field(default_factory=TestChild, description="child of parent")
    # TestChild()
    # : TestChild = Field(description="The child item")

if __name__ == '__main__':
    output = {'car_id': 5.15,
              "car_make": [{"car_brand": "BMC",
                            "color": "red",
                            "weird": 12}, {"car_brand": "BMC",
                            "color": "Green",
                            "weird": 12}]}
    output = {
              "city": "San Diego",
              "coordinates": {
                "longitude": -117.215935,
                "latitude": 32.873055
              },
              "country": "United States",
              "name": "San Diego Marriott La Jolla"
            }
    output = {
    'search_key': 'eyJkcml2ZXJzQWdlIjozMCwiZHJvcE9mZkRhdGVUaW1lIjoiMjAyNS0wNS0yNlQwODowMDowMCJ9',
    'title': 'Car rentals',
    'content': {
        'filters': {'countLabel': '2 results'},
        'items': [
            {'content': {'contentType': 'carsSearchResultsSecondaryPromotional'}, 'positionInList': 1, 'type': 'SHELL_REGION_VIEW'}
        ]
    },
    'type': 'cars',
    'meta': {'response_code': 200},
    'search_results': [
        {
            'vehicle_id': '123456',
            'route_info': {
                'pickup': {
                    'name': 'Sample Airport',
                    'address': '123 Sample St, Sample City, USA',
                    'latitude': '12.3456',
                    'longitude': '-65.4321',
                    'location_type': 'SHUTTLE_BUS'
                },
                'dropoff': {
                    'name': 'Sample Airport',
                    'address': '123 Sample St, Sample City, USA',
                    'latitude': '12.3456',
                    'longitude': '-65.4321',
                    'location_type': 'SHUTTLE_BUS'
                }
            },
            'rating_info': {
                'average': 8.1,
                'value_for_money': 7.5,
                'cleanliness': 8.3
            },
            'vehicle_info': {
                'v_name': 'Sample Car Model',
                'group': 'Economy',
                'seats': '4',
                'doors': '4',
                'fuel_policy': 'Like for like',
                'transmission': 'Automatic'
            },
            'supplier_info': {
                'name': 'Sample Supplier',
                'address': '123 Supplier Rd, Supplier City, USA',
                'logo_url': 'https://example.com/logo.png'
            },
            'pricing_info': {
                'price': 45.99,
                'currency': 'USD'
            }
        },
        {
            'vehicle_id': '7891011',
            'route_info': {
                'pickup': {
                    'name': 'Another Airport',
                    'address': '456 Another St, Another City, USA',
                    'latitude': '23.4567',
                    'longitude': '-54.3210',
                    'location_type': 'SHUTTLE_BUS'
                },
                'dropoff': {
                    'name': 'Another Airport',
                    'address': '456 Another St, Another City, USA',
                    'latitude': '23.4567',
                    'longitude': '-54.3210',
                    'location_type': 'SHUTTLE_BUS'
                }
            },
            'rating_info': {
                'average': 7.2,
                'value_for_money': 6.8,
                'cleanliness': 7.5
            },
            'vehicle_info': {
                'v_name': 'Another Car Model',
                'group': 'Compact',
                'seats': '5',
                'doors': '4',
                'fuel_policy': 'Full to full',
                'transmission': 'Manual'
            },
            'supplier_info': {
                'name': 'Another Supplier',
                'address': '456 Supplier Ave, Supplier Town, USA',
                'logo_url': 'https://example.com/another_logo.png'
            },
            'pricing_info': {
                'price': 55.49,
                'currency': 'USD'
            }
        }
    ],
    'sort': [
        {'identifier': 'recommended', 'name': 'Recommended – best first'},
        {'identifier': 'price_low_to_high', 'name': 'Price - lowest first'}
    ],
    'filter': [
        {'id': 'carCategory', 'title': 'Car Type', 'categories': [
            {'id': 'small', 'name': 'Small', 'count': 5},
            {'id': 'medium', 'name': 'Medium', 'count': 10}
        ]},
        {'id': 'pricePerDayBuckets', 'title': 'Price per day', 'categories': [
            {'id': 'bucket_1', 'name': 'US$0 - US$50', 'count': 3},
            {'id': 'bucket_2', 'name': 'US$50 - US$100', 'count': 7}
        ]}
    ],
    'search_context': {
        'searchId': 'abc123-search-id',
        'searchKey': 'eyJkcml2ZXJzQWdlIjozMCwiZHJvcE9mZkRhdGVUaW1lIjoiMjAyNS0wNS0yNlQwODowMDowMCJ9'
    },
    'provider': 'rentalcars'
}
    # output = {'car_id': 5,
    #           'car_colors': ['red', 'blue']}

    
    # output = {'car_id': 5}
    # output = {'car_id': 5,
    #           "car_make": {"car_brand": "BMC",
    #                         "color": "Green"}}
    # type_repr = parse_output(output, "bob")
    # type_repr = parse_output_to_basemodel(output, "bob")

    open_ai_json_parameter = {"type": "object",
        "properties": {
          "pick_up_longitude": {
            "type": "number",
            "description": "The pick up location's `longitude`. `pick_up_longitude` can be retrieved from `api/v1/cars/searchDestination`**(Search Car Location)** endpoint in **Car Rental** collection as `longitude` inside `coordinates` object."
          },
          "drop_off_time": {
            "type": "string",
            "description": "Drop off time\nFormat: **HH:MM**\n*Note: The format of time is 24 hours.* TIME (24-hour HH:MM)"
          },
          "drop_off_latitude": {
            "type": "number",
            "description": "The drop off location's `latitude`. `drop_off_latitude` can be retrieved from `api/v1/cars/searchDestination`**(Search Car Location)** endpoint in **Car Rental** collection as `latitude` inside `coordinates` object."
          },
          "drop_off_date": {
            "type": "string",
            "description": "Drop off date\nFormat: **YYYY-MM-DD**DATE (YYYY-MM-DD)"
          },
          "pick_up_time": {
            "type": "string",
            "description": "Pick up time\nFormat: **HH:MM**\n*Note: The format of time is 24 hours.* TIME (24-hour HH:MM)"
          },
          "pick_up_latitude": {
            "type": "number",
            "description": "The pick up location's `latitude`. `pick_up_latitude` can be retrieved from `api/v1/cars/searchDestination`**(Search Car Location)** endpoint in **Car Rental** collection as `latitude` inside `coordinates` object."
          },
          "pick_up_date": {
            "type": "string",
            "description": "Pick up date\nFormat: **YYYY-MM-DD**DATE (YYYY-MM-DD)"
          },
          "drop_off_longitude": {
            "type": "number",
            "description": "The drop off location's `longitude`. `drop_off_longitude` can be retrieved from `api/v1/cars/searchDestination`**(Search Car Location)** endpoint in **Car Rental** collection as `longitude` inside `coordinates` object."
          }
        }}
    # output = {
    #     "awef": [1,2.1,3,4]
    # }
    # type_dict = parse_output_to_dict(output, "")
    import json
    with open("some_data.json", "r") as f:
        output = json.load(f)
    type_repr = parse_output_to_basemodel(output,  "")

    small_obj = {"awef": "a",
                 "bawef": "b",
                 "cawef": {"dawef": "a",
                           "cawef": "fawef"}}
    # type_repr = parse_output_to_basemodel(output, "bob")

    # awef = TestParent()
    # leaves_descriptions = get_leaves_of_basemodel(type_repr)