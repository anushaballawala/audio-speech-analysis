import parselmouth

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme() # Use seaborn's default style to make attractive graphs
plt.rcParams['figure.dpi'] = 100 # Show nicely large images in this notebook

snd = parselmouth.Sound("raw_audio/testsoundmono.mp3")

plt.figure()
plt.plot(snd.xs(), snd.values.T)
plt.xlim([snd.xmin, snd.xmax])
plt.xlabel("time [s]")
plt.ylabel("amplitude")
plt.savefig("figures/normalsound.png")
plt.show() # or plt.savefig("sound.png"), or plt.savefig("sound.pdf")


