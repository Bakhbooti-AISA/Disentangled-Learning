from pycatch22 import catch22_all
import numpy as np

x = np.random.randn(200)
out = catch22_all(x)

print(len(out["values"]))
print(out["names"][:5])
