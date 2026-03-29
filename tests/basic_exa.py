from exa_py import Exa
from dotenv import load_dotenv
import os

load_dotenv()

exa = Exa(os.getenv("EXA_API_KEY"))

result = exa.search(
  "Latest news on Nvidia",
  num_results = 10,
  type = "auto",
  contents = {
    "highlights": {
      "max_characters": 4000
    }
  }
)

print(result)