from agent.agent import graph, functionDatabase
from agent.state import State

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import time

# Setup LangChain
# llm = OpenAI(model="gpt-3.5-turbo")  # replace with your local/custom model
# prompt = PromptTemplate.from_template("Answer this: {question}")
# chain = LLMChain(llm=llm, prompt=prompt)


app = FastAPI()

# @app.post("/v1/chat/completions")
# async def chat_completions(request: Request):
#     payload = await request.json()
#     messages = payload.get("messages", [])
#     user_message = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")

#     # Run through LangChain
#     user_input: State = { "plan": plan,
#                     "mode": PLANNING,
#                     "plan_idx": 0,
#                     "results_cache": {},
#                     "messages": [{"role": "user", "content": user_message}]}
    
#     result: State = graph.invoke(user_input)
#     response_message = result["messages"][-1]
#     # chain.run(question=user_message)

#     return JSONResponse({
#         "id": f"chatcmpl-{int(time.time())}",
#         "object": "chat.completion",
#         "created": int(time.time()),
#         "model": "local-custom-model",
#         "choices": [{
#             "index": 0,
#             "message": response_message,
#             "finish_reason": "stop"
#         }],
#         "usage": {
#             "prompt_tokens": 0,
#             "completion_tokens": 0,
#             "total_tokens": 0
#         }
#     })

def convert_message(message):
    if message.type == "human":
        return {"type": message.type, 
                           "content": message.content,
                           }
    elif message.type == "ai":
        return  {"type": message.type, 
                    "content": message.content,
                    "tool_calls": message.tool_calls
                    }
    elif message.type == "tool":
        return {"type": message.type, 
                    "content": message.content,
                    "name": message.name
                    }

@app.post("/invoke")
async def invoke_custom(request: Request):
    # try:
    payload = await request.json()
    input_data = payload.get("input", "")
    config_data = payload.get("config", "")
    tools = payload.get("tools", None)

    functionDatabase.set_tool_limitations(tools)
    result = graph.invoke(input_data, config=config_data)

    # convert messages
    result["messages"] = [convert_message(message) for message in result["messages"]]
    result["results_cache"] = {}
    return JSONResponse({
        "status": "success",
        "output": result
    })
    # except Exception as e:
    #     return JSONResponse({
    #         "status": "failed",
    #         "output": str(e)
    #     })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)