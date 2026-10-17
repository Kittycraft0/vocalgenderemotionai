"""
Alias / Forwarder for main13soundreconstruction.py
Allows running either:
    python main13soundreconstruction.py
or
    python main13sounreconstruction.py
"""
import sys
import os

# Import and run the main reconstruction module
import main13soundreconstruction

if __name__ == "__main__":
    main13soundreconstruction.main()

