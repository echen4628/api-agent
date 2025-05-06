from utils.parse_functions import parse_output_to_basemodel, parse_inputs_to_basemodel, get_leaves_of_basemodel

class FunctionWrapper:
    def __init__(self):
        self.name = ""
        self.description = ""
        self.parameters = None # some kind of pydantic baseclass
        self.required = list()
        self.output = None # some kind of pydantic baseclass
        self.input_dependencies = {}
        self.parameter_leaves = list()
        self.output_leaves = list()
    
    def parse(self, open_ai_json):
        self.name = open_ai_json["name"]
        self.description = open_ai_json["description"]
        self.parameters = parse_inputs_to_basemodel(open_ai_json["parameters"], f"{self.name}_input") # maybe this can be a helper function
        import pdb; pdb.set_trace()
        self.required = open_ai_json["parameters"]["required"]
        # get the leaves of self.parameters
        self.parameter_leaves = get_leaves_of_basemodel(self.parameters)
        self.input_dependencies = {leaf:[] for leaf in self.parameter_leaves}
    
    def parse_output(self, sample_output):
        self.output = parse_output_to_basemodel(sample_output, f"{self.name}_output")
        self.output_leaves = get_leaves_of_basemodel(self.output)

if __name__ == '__main__':
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