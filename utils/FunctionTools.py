from utils.FunctionWrapper import FunctionWrapper
from typing import List, Union, Dict
import os
import faiss
from langchain_openai import OpenAIEmbeddings    
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import OpenAIEmbeddings
import json

from langchain_core.documents import Document
import logging
import jsonpickle
import pickle

# Configure the logger to output debug messages to the console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a logger instance
logger = logging.getLogger(__name__)

class FunctionDependencyGraph:
    def __init__(self):
        self.functions:List[List[FunctionWrapper]] = []

class DependencyEdge:
    def __init__(self, destination, destination_func_name, source, source_func_name):
        self.destination = destination
        self.destination_func_name = destination_func_name
        self.source = source
        self.source_func_name = source_func_name
    
    def __str__(self):
        return f"Use output {self.source} from function {self.source_func_name} as the input for {self.destination} of function {self.destination_func_name}"

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
    def __init__(self, embedding_name, inputs_desc_vector_store_path,
                        outputs_desc_vector_store_path, func_desc_vector_store_path,
                        name_to_function_json_path):
        
        self.embeddings = OpenAIEmbeddings(model=embedding_name)
        self.inputs_desc_vector_store_path = inputs_desc_vector_store_path
        self.outputs_desc_vector_store_path = outputs_desc_vector_store_path
        self.func_desc_vector_store_path = func_desc_vector_store_path
        self.name_to_function_json_path = name_to_function_json_path
        for vector_store_path in [inputs_desc_vector_store_path, outputs_desc_vector_store_path, func_desc_vector_store_path]:
            if not os.path.exists(vector_store_path):
                logger.warning(f"Cannot find vector store at {vector_store_path}")
                logger.info(f"Creating vector store at {inputs_desc_vector_store_path}...")
                index = faiss.IndexFlatL2(len(self.embeddings.embed_query("hello world")))

                vector_store = FAISS(
                    embedding_function=self.embeddings,
                    index=index,
                    docstore=InMemoryDocstore(),
                    index_to_docstore_id={},
                )
                vector_store.save_local(vector_store_path)

        self.inputs_desc_vector_store = FAISS.load_local(inputs_desc_vector_store_path, self.embeddings, allow_dangerous_deserialization=True)
        self.outputs_desc_vector_store = FAISS.load_local(outputs_desc_vector_store_path, self.embeddings, allow_dangerous_deserialization=True)
        self.func_desc_vector_store = FAISS.load_local(func_desc_vector_store_path, self.embeddings, allow_dangerous_deserialization=True)
        
        self.name_to_function: Dict[str, FunctionWrapper] = {}
        if os.path.exists(name_to_function_json_path):
            with open(name_to_function_json_path, "r") as f:
                written_instance = f.read()
                self.name_to_function = jsonpickle.decode(written_instance)
            # with open(name_to_function_json_path, "rb") as f:
            #     self.name_to_function = pickle.load(f)
        self.tool_limitations = None

    def set_tool_limitations(self, tools: Union[List[str], None]):
        self.tool_limitations = tools
    
    def add_function(self, function: FunctionWrapper):
        # import pdb; pdb.set_trace()
        if function.name in self.name_to_function:
            logger.debug(f"function {function.name} is already in the database, skipping!")
            return
        self.name_to_function[function.name] = function

        func_desc_document = Document(
            page_content=function.description,
            metadata={"source": function.name},
        )
        self.func_desc_vector_store.add_documents(documents=[func_desc_document])

        if function.parameter_leaves:
            parameter_desc_documents = []
            for parameter_leaf in function.parameter_leaves:
                parameter_desc_document = Document(
                    page_content=function.parameter_leaves[parameter_leaf],
                    metadata={"source": parameter_leaf, "function": function.name},
                )
                parameter_desc_documents.append(parameter_desc_document)
            self.inputs_desc_vector_store.add_documents(documents=parameter_desc_documents)
        else:
            logger.warning(f"function {function.name} does not have input parameters.")

        if function.output_leaves:
            outputs_desc_documents = []
            for output_leaf in function.output_leaves:
                outputs_desc_document = Document(
                    page_content=function.output_leaves[output_leaf],
                    metadata={"source": output_leaf, "function": function.name},
                )
                outputs_desc_documents.append(outputs_desc_document)
            self.outputs_desc_vector_store.add_documents(documents=outputs_desc_documents)

        else:
            logger.warning(f"function {function.name} does not have outputs.")
        if not os.path.exists(self.name_to_function_json_path):
            os.makedirs( os.path.dirname(self.name_to_function_json_path))
        with open(self.name_to_function_json_path, 'w') as f:
            f.write(jsonpickle.encode(self.name_to_function))
        # with open(self.name_to_function_json_path, 'wb') as f:
        #     pickle.dump(self.name_to_function, f)
        self.func_desc_vector_store.save_local(self.func_desc_vector_store_path)
        self.inputs_desc_vector_store.save_local(self.inputs_desc_vector_store_path)
        self.outputs_desc_vector_store.save_local(self.outputs_desc_vector_store_path)

    
    def search(self, query:str) -> Union[str, FunctionWrapper]:
        '''
        Finds the name of a function to perform the query
        '''
        # first search by function name
        if query in self.name_to_function:
            functions = [self.name_to_function[query]]
        else:
            # import pdb; pdb.set_trace()
            if self.tool_limitations:
                retriever = self.func_desc_vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3, "filter": {"source": {"$in": self.tool_limitations}}})
            else:
                retriever = self.func_desc_vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3})

            # then search each of the inputs and outputs and description to see if there are any similar functions
            desc_res = retriever.invoke(query)

            # if self.tool_limitations:
            #     desc_res = retriever.invoke(query, filter={"source": {"$in": self.tool_limitations}})
            # else:
            #     desc_res = retriever.invoke(query)
            functions = [self.name_to_function[desc.metadata["source"]] for desc in desc_res]
            # desc_res = self.func_desc_vector_store.search(query)
            # # maybe extract the function name and the value match, so that this would be a Tuple
            # input_res = self.inputs_desc_vector_store.search(query)
            # output_res = self.outputs_desc_vector_store.search(query)

        return "\n\n".join([self.convert_function_desc_to_str(function) for function in functions])

    def convert_function_desc_to_str(self, function: FunctionWrapper, include_dependencies=False):
        base_string = f"Function {function.name}:\n -{function.description}.\n -inputs: {function.parameter_leaves}"
        if include_dependencies:
            # import pdb; pdb.set_trace()
            input_resolutions = {name: [edge.__str__() for edge in  dependency_edge] for name, dependency_edge in function.input_dependencies.items()}
            base_string += f"\n -potential input resolutions: {input_resolutions}"
        return base_string
    
    def find_dependency(self, function_names: List[str]) -> Union[List[Union[str, FunctionWrapper]], str]:
        '''
        Applies a similarity search to find functions that could be used to provide dependencies for a list of functions with unknown dependencies.

        Note that some function dependencies should instead be obtained from the user.
        '''
        try:
            functions = [self.name_to_function[name] for name in function_names]
        except KeyError as ke:
            return f"{ke} is not a valid function name. Try something explicitly labeled as a function name. You may be confused by information in the function description."
        except Exception as e:
            return str(e)

        for function in functions:
            if self.tool_limitations:
                retriever = self.outputs_desc_vector_store.as_retriever(search_type="mmr",
                            search_kwargs={"k": 3, "filter":{"function": {"$neq": function.name,
                                                                          "$in": self.tool_limitations} }})
            else:
                retriever = self.outputs_desc_vector_store.as_retriever(search_type="mmr",
                            search_kwargs={"k": 3, "filter":{"function": {"$neq": function.name} }})
            for parameter, parameter_desc in function.parameter_leaves.items():
                output_res = retriever.invoke(parameter_desc)
                dependency_edges = [DependencyEdge(destination=parameter, destination_func_name=function.name,
                                            source= res.metadata["source"], source_func_name=res.metadata["function"]) for res in output_res]
                function.input_dependencies[parameter] = dependency_edges
        return [self.convert_function_desc_to_str(function, include_dependencies=True) for function in functions]
    # def find_dependency(self, functions: List[FunctionWrapper]) -> List[FunctionWrapper]:
    #     for function in functions:
    #         retriever = self.outputs_desc_vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3, "filter":{"function": {"$neq": function.name} }})
    #         for parameter, parameter_desc in function.parameter_leaves.items():
    #             output_res = retriever.invoke(parameter_desc)
    #             import pdb; pdb.set_trace()
    #             dependency_edges = [DependencyEdge(destination=parameter, destination_func_name=function.name,
    #                                         source= res.metadata["source"], source_func_name=res.metadata["function"]) for res in output_res]
    #             function.input_dependencies[parameter] = dependency_edges
    #     return functions

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
                for next_function in most_recent_function.input_dependencies.items():
                    assert len(next_function) == 1
                    if next_function[0].source_function_name != "get_from_user":
                        next_level.append(self.database.name_to_function(next_function[0].source_function_name))
            if next_level != []:
                graph.functions.append(next_level)
            else:
                break
        return graph        
    
        