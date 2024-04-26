# Setting the matplotlib backend to something other than Tk, as it causes issues when using
# Tkinter GUIs and plots together. Tkinter is not thread-safe so when you close a plot it ends
# up blowing up the GUI and deleting its thread. This is evidently the solution:
#
#     https://stackoverflow.com/questions/27147300

import matplotlib

# TODO commented out because you have to do this at the very first import, and it's unclear
# where that's happening. Don't need this unless using Tkinter GUIs.

# matplotlib.use('Agg')
