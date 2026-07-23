# %% [markdown]
# ## Experimente
# 
# Verändert jeweils nur einen Parameter:
# 
# ### 1. Mehr Schichten
# Vergleiche:
# - `784 → 64 → 10`
# - `784 → 512 → 256 → 128 → 10`
# 
# Frage:
# - Wird das Modell besser?
# - Wie verändert sich die Trainingszeit?
# 
# 
# ### 2. Aktivierungsfunktion
# 
# Vergleiche:
# 
# - ReLU
# - Sigmoid
# - Tanh
# 
# Frage:
# - Warum funktioniert ReLU meistens besser?
# 
# 
# ### 3. Dropout
# 
# Vergleiche:
# - `dropout_rate=0.0`
# - `dropout_rate=0.5`
# 
# Frage:
# - Warum kann ein schlechteres Trainingsergebnis trotzdem ein besseres Modell ergeben?
# 
# 
# ### 4. Lernrate
# 
# Teste:
# - `learning_rate = 0.0001`
# - `learning_rate = 0.001`
# - `learning_rate = 0.01`
# 
# Frage:
# - Was passiert bei zu kleinen oder zu großen Schritten?
# 

# %% [markdown]
# ## setup

# %%


# %% [markdown]
# ## Experiments

# %%


# %% [markdown]
# ### Results compare

# %%


# %% [markdown]
# * more layers → deeper learning
# * activation functions → nonlinearity
# * dropout → generalization
# * learning rate → gradient descent behavior.


