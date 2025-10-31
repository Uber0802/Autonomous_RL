import numpy as np

value_og = np.load("/workspace/Autonomous_RL/SimplerEnv/bottle_shovel-320-train_twice-joint-reset-seed_2.npy")
value = np.load("/workspace/Autonomous_RL/SimplerEnv/bottle_shovel-320-train_twice-joint-reset-seed_2_embed.npy")

print(len(value_og))
print(value.shape)
diff = value - value_og

print(f"{np.mean(np.abs(diff)):.4f}")
print(f"{np.mean(diff):.4f}")
print(f"{np.max(diff):.4f}")
print(f"{np.min(diff):.4f}")