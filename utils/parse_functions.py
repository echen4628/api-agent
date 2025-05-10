from pydantic import BaseModel, create_model, Field
from pydantic.fields import FieldInfo
from typing import TypedDict, List, Annotated, Dict
from copy import deepcopy
from utils.constants import type_mapping
# def parse_output(output):
#     if type(output) == list:
#         return [parse_output(output[0])]
#     elif type(output) != dict:
#         return type(output)
#     else:
#         type_repr = {}
#         for key in output.keys():
#             type_repr[key] = parse_output(output[key])
#         return type_repr

def parse_output(output, name):
    if type(output) == list:
        return list[parse_output(output[0], name)]
    elif type(output) != dict:
        return type(output)
    else:
        type_repr = {}
        for key in output.keys():
            type_repr[key] = deepcopy(parse_output(output[key], str(key)))
        return TypedDict(name, type_repr)
    
def parse_output_to_basemodel(output, name):
    if isinstance(output, list):
        # Assume homogeneous list
        inner = parse_output_to_basemodel(output[0], name + "Item")
        return List[inner]
    elif not isinstance(output, dict):
        return type(output)
    else:
        fields = {}
        for key, value in output.items():
            field_type = parse_output_to_basemodel(value, name + str(key).capitalize())
            fields[key] = (field_type, ...)
        return create_model(name, **fields)

def parse_inputs_to_basemodel(open_ai_json_parameter, name):
    fields = {}
    for key, value in open_ai_json_parameter["properties"].items():
        fields[key] = Annotated[type_mapping[value['type']], Field(description=value['description'])]
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
        elif obj.annotation in [str, int, float, bool]:
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
        import pdb; pdb.set_trace()

    return leaf_descriptions

class TestChild(BaseModel):
    child_field: str = Field(default="", description="this is the child field")
  
class TestParent(BaseModel):
    my_child: TestChild = Field(default_factory=TestChild, description="child of parent")
    # TestChild()
    # : TestChild = Field(description="The child item")

if __name__ == '__main__':
    output = {'car_id': 5,
              "car_make": [{"car_brand": "BMC",
                            "color": "Green"}, {"car_brand": "BMC",
                            "color": "Green"}]}
    
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
    type_repr = parse_output_to_basemodel(output, "bob")

    # awef = TestParent()
    leaves_descriptions = get_leaves_of_basemodel(type_repr)