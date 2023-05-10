## Installation
Install Python 3.8+, then run:
```bash
pip install -r requirements.txt
```

## Running
```bash
python main.py
```

NOTE: The generation of a powerlaw tree is non-deterministic and may not succeed
 within the default number of attempts.

By default, the script will attempt to re-generate the powerlaw tree indefinitely
 until it succeeds, which is guaranteed to occur. However, you can also fix this
 by removing the powerlaw tree tuple entry from `gen_tops()`. This is a constraint
 of NetworkX's implementation of powerlaw tree generation.

The constants declared in `ALL_CAPS` in the "main" function at the bottom of the
 script can be changed to enable graph visualization and modify the number of nodes
 used and the amount of energy at each node.