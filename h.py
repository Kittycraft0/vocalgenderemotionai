# 10/30/2025
import numpy as np
import matplotlib.pyplot as plt

k=np.array([1,2,3])

p0=1/2

p=np.array([1/2,1/2,1/2])
p0 = np.empty((0, len(p)))

print(f"0: {p}")
    
for i in range(30):
    p=k*p*(1-p)
    print(f"{i}: {p}")
    p0=np.vstack([p0,p])



plt.subplot(4, 1, 3)
print(f"Sequence")
#print(rms_waveform)
plt.plot(p0, label="iteration", color='orange')
#plt.plot(np.linspace(0, len(energy), len(new_energy)), new_energy, label="Processed STE", color='blue')
plt.title("Population over time")
plt.xlabel("# Iterations")
plt.ylabel("Number")
plt.legend()
plt.tight_layout()
plt.show()


