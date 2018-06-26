#!/usr/bin/python

import argparse
from datetime import datetime
import json

import pyhmf as pynn
import pymarocco

from pysthal.command_line_util import init_logger
init_logger("WARN", [])

class LayeredFeedforwardNetwork(object):
    def __init__(self, num_layers, conn_prob, neurons_per_layer, marocco, model=pynn.EIF_cond_exp_isfa_ista):
        self.neurons_per_layer = neurons_per_layer
        self.num_layers = num_layers 
        self.conn_prob = conn_prob
        self.model = model
        self.marocco = marocco

        pynn.setup(marocco=self.marocco)

    def build(self):

        self.neurons = []
        for i in range(self.num_layers):
            self.neurons.append(pynn.Population(self.neurons_per_layer, self.model))

        connector = pynn.FixedProbabilityConnector(
                p_connect=self.conn_prob,
                allow_self_connections=False,
                weights=0.003)
        proj = []
        for i in range(1, self.num_layers):
            proj = pynn.Projection(
                self.neurons[i-1],
                self.neurons[i],
                connector,
                target='excitatory',
                rng=pynn.NativeRNG(42))

    def run(self):
        pynn.run(1)
        pynn.end()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--conn_prob', default = 1., type=float)
    parser.add_argument('--neurons_per_layer', default = 100, type=int)

    args = parser.parse_args()

    marocco = pymarocco.PyMarocco()
    marocco.continue_despite_synapse_loss = True
    marocco.calib_backend = pymarocco.PyMarocco.CalibBackend.Default

    start = datetime.now()
    r = LayeredFeedforwardNetwork(args.num_layers, args.conn_prob, args.neurons_per_layer, marocco)
    r.build()
    mid = datetime.now()
    r.run()
    end = datetime.now()

    result = {
        "model" : "random_network",
        "num_layers": args.num_layers,
        "conn_prob": args.conn_prob,
        "neurons_per_layer": args.neurons_per_layer,
        "task" : "num_layers{}_conn_prob{}_neurons_per_layer{}".format(args.num_layers,\
                                                 args.conn_prob, args.neurons_per_layer),
        "timestamp" : datetime.now().isoformat(),
        "results" : [
            {"type" : "performance",
             "name" : "setup_time",
             "value" : (end-mid).total_seconds(),
             "units" : "s",
             "measure" : "time"
         },
            {"type" : "performance",
             "name" : "total_time",
             "value" : (end-start).total_seconds(),
             "units" : "s",
             "measure" : "time"
         },
            {"type" : "performance",
             "name" : "synapses",
             "value" : marocco.stats.getSynapses()
         },
            {"type" : "performance",
             "name" : "neurons",
             "value" : marocco.stats.getNumNeurons()
         },
            {"type" : "performance",
             "name" : "synapse_loss",
             "value" : marocco.stats.getSynapseLoss()
         },
            {"type" : "performance",
             "name" : "synapse_loss_after_l1",
             "value" : marocco.stats.getSynapseLossAfterL1Routing()
         }
        ]
    }

    with open("random_num_layers{}_conn_prob{}neurons_per_layer{}_results.json".format(args.num_layers,\
                                                   args.conn_prob, args.neurons_per_layer), 'w') as outfile:
        json.dump(result, outfile)

if __name__ == '__main__':
    main()
