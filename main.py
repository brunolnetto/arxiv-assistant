from os import getenv
from dotenv import load_dotenv

load_dotenv()

from display import interact_with_graph
from graph import graph 

if __name__ == "__main__":
    interact_with_graph(graph)
