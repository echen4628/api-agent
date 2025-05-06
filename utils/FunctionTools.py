from utils.FunctionWrapper import FunctionWrapper
from typing import List, Union

class FunctionDependencyGraph:
    def __init__(self):
        self.functions:List[List[FunctionWrapper]] = []

class DependencyEdge:
    def __init__(self, destination, destination_func_name, source, source_func_name):
        self.destination = destination
        self.destination_func_name = destination_func_name
        self.source = source
        self.source_func_name = source_func_name

class FunctionDependencyReducer:
    def __init__(self):
        self.llm = None
    
    def reduce(self, graph: FunctionDependencyGraph, purpose: str) -> FunctionDependencyGraph:
        # potentially make this, create a copy so that it is not inplace
        functions_to_reduce = graph[-1]
        for func in functions_to_reduce:
            # can potentially pass the whole thing
            for i in range(len(func.input_dependencies)):
                input_dependency = func.input_dependencies[i]
                resolved_input_dependency = self.llm.invoke(input_dependency)
                func.input_dependencies[i] = resolved_input_dependency
        return graph

class FunctionDatabase:
    def __init__(self):
        self.inputs_desc_vector_store = None
        self.output_desc_vector_store = None
        self.func_desc_vector_store = None
        self.name_to_function = {}
    
    def add_function(self, function: FunctionWrapper):
        self.name_to_function[function.name] = function

        # maybe these vector stores should all be combined?
        self.inputs_desc_vector_store.insert(function.parameters)
        self.output_desc_vector_store.insert(function.output)
        self.func_desc_vector_store.insert(function.description)
    
    def search(self, query) -> Union[str, FunctionWrapper]:
        # first search by function name
        if query in self.name_to_function:
            return self.name_to_function[query]
        else:
            # then search each of the inputs and outputs and description to see if there are any similar functions
            desc_res = self.func_desc_vector_store.search(query)
            # maybe extract the function name and the value match, so that this would be a Tuple
            input_res = self.inputs_desc_vector_store.search(query)
            output_res = self.outputs_desc_vector_store.search(query)
            return f"No functions are exactly named query, here are the closest functions: {desc_res}, {input_res}, {output_res}."

    def find_dependency(self, functions: List[FunctionWrapper]) -> List[FunctionWrapper]:
        for function in functions:
            for parameter in function.parameter_leaves:
                output_res = self.output_desc_vector_store.search(parameter)
                # TODO: need to put the parameter name and function name in the metadata of the documents
                dependency_edges = [DependencyEdge(destination=parameter, destination_func_name=function.name,
                                            source= res.parameter_name, source_func_name=res.function_name) for res in output_res]
                function.input_dependencies[parameter] = dependency_edges
        return functions

class Planner:
    def __init__(self, function_database: FunctionDatabase, dependency_reducer: FunctionDependencyReducer, max_depth: int):
        self.database = function_database
        self.reducer = dependency_reducer
        self.max_depth = max_depth
    
    def plan_function_call(self, goal_function: FunctionWrapper, purpose: str) -> FunctionDependencyGraph:
        # the goal_function could just be the function name
        graph = FunctionDependencyGraph()
        graph.functions.append([goal_function])
        while len(graph.functions) <= self.max_depth:
            graph.functions[-1] = self.database.find_dependency(graph.functions[-1])
            graph = self.reducer.reduce(graph, purpose)
            # extend it
            next_level = []
            for most_recent_function in graph.functions[-1]:
                for next_function in most_recent_function.input_dependencies.values():
                    assert len(next_function) == 1
                    if next_function[0].source_function_name != "get_from_user":
                        next_level.append(self.database.name_to_function(next_function[0].source_function_name))
            if next_level != []:
                graph.functions.append(next_level)
            else:
                break
        return graph        
    
        