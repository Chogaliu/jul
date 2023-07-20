import numpy as np
from gpcm import RGPCM

model = RGPCM(window=2, scale=1, n_u=30, t=(0,10))

t = np.linspace(0,10,100)
K,y = model.sample(t)

model.fit(t, y)

print(model.elbo(t, y))

posterior = model.condition(t, y)
mean, var = posterior.predict(t)
