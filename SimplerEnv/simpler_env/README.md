# Tasks
1. `OneObjectTwoReceptacle-v1`
2. `TwoObjectOneReceptacle-v1`
3. `TwoObjectTwoReceptacle-v1`

# Environment Functions
- All function are in `SimplerEnv/simpler_env/env/simpler_wrapper.py`
- All examples using `TwoObjectTwoReceptacle-v1`, seed=0, num_envs=2
- `runner.env.reset()` reset task to `put watering can on yellow_plate`

## Get task instructions in all envs
```bash
print(runner.env.get_language_instruction())

# Output
['put watering can on yellow_plate', 'put watering can on yellow_plate']
```

## Get task pool in all env
```bash
print(runner.env.get_task_pool())

# Output
[
    ['put watering can on yellow_plate','put watering can on envelope', 'put ketchup bottle on yellow_plate', 'put ketchup bottle on envelope'], 
    ['put watering can on yellow_plate', 'put watering can on envelope', 'put ketchup bottle on yellow_plate', 'put ketchup bottle on envelope']
]
```

## Get object (carrot) names in all envs
```bash
print(runner.env.get_object_names())

# Output
[['watering can', 'ketchup bottle'], ['watering can', 'ketchup bottle']]
```

## Get receptacle (plate) names in all envs
```bash
print(runner.env.get_receptacle_names())

# Output
[['yellow_plate', 'envelope'], ['yellow_plate', 'envelope']]
```

## Change task
Evaluation will also change.
```bash
runner.env.set_task(['ketchup bottle', 'ketchup bottle'], ['envelope', 'envelope'])
print(runner.env.get_language_instruction())

# Output
['put ketchup bottle on envelope', 'put ketchup bottle on envelope']
```