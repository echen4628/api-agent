import requests
from utils.constants import PLANNING

# url = 'http://0.0.0.0:8000'
url = 'http://127.0.0.1:8000/invoke'
myobj = {"config": {"configurable": {"thread_id": "1"}},
         "input": { "plan": [],
                    "mode": PLANNING,
                    "plan_idx": 0,
                    "results_cache": {},
                    "messages": [{"role": "user", "content": "Today is October 13th, 2024. I want to rent a car for a day at the San Diego Marriott La Jolla. Could you compare the price differences for picking up the car at 8 AM tomorrow and the day after tomorrow at the same place for a 24-hour rental?"}]}
}


x = requests.post(url, json = myobj)

# print(x.text)


# if __name__ == "__main__":
#     config = {"configurable": {"thread_id": "1"}}

#     # def stream_graph_updates(user_input: str):
#     #     for events in graph.stream({"messages": [{"role": "user", "content": user_input}]},
#     #                                config,
#     #                                stream_mode="updates"):
#     #         for value in events.values():
#     #             print("Assistant:", value["messages"][-1].content)

#     def stream_graph_updates(user_input: State):
#         for events in graph.stream(user_input,
#                                    config,
#                                    stream_mode="updates"):
#             for value in events.values():
#                 print("Assistant:", value["messages"][-1].content)

#     plan = []

#     while True:
#         user_input_message = input("User: ")
#         if user_input_message.lower() in ["quit", "exit", "q"]:
#             print("Goodbye!")
#             break
#         user_input: State = { "plan": plan,
#                         "mode": PLANNING,
#                         "plan_idx": 0,
#                         "results_cache": {},
#                         "messages": [{"role": "user", "content": user_input_message}]}

#         stream_graph_updates(user_input)
