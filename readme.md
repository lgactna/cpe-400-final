## Installation
Install Python 3.8+, then run:
```bash
pip install -r requirements.txt
```

## Running
```bash
python main.py
```

NOTE: Running all simulations in __main__ may fail, since the generation of a powerlaw
tree is non-deterministic and may not succeed within the default number of attempts.

To fix this, remove the Powerlaw Tree tuple entry from `topologies` in main(), or 
re-run the script until generation succeeds. This is a constraint of NetworkX.
