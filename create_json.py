import json

benchmark = {"model": {
                       "name": "layered_feedforward",
                       "description": "mapping a ff network on the Brainscales wafer"
             },
             "tasks": []}

for num_layers in range(2,11):
    benchmark['tasks'].append({
                       "name": "layered_feedfoward_network_num_layers{}".format(num_layers),
                       "command": "mapping/networks/layered_feedforward/run.py --num_layers {}".format(num_layers)
    })

with file("benchmarks.json", "w") as outfile:
    json.dump([benchmark], outfile, indent=4)      
