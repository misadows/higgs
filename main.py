import argparse
import os

import train_rhorho

types = {"nn_rhorho": train_rhorho.start}

parser = argparse.ArgumentParser(description='Train classifier')
parser.add_argument("-t", "--type", dest="TYPE", choices=types.keys(), default='nn_rhorho')
parser.add_argument("-l", "--layers", dest="LAYERS", type=int, help = "number of NN layers", default=6)
parser.add_argument("-s", "--size", dest="SIZE", type=int, help="NN size", default=100)
parser.add_argument("-lambda", "--lambda", type=float, dest="LAMBDA", help="value of lambda parameter", default=0.0)
parser.add_argument("-m", "--method", dest="METHOD", choices=["A", "B", "C"], default="A")
parser.add_argument("-o", "--optimizer", dest="OPT", 
	choices=["GradientDescentOptimizer", "AdadeltaOptimizer", "AdagradOptimizer",
	 "ProximalAdagradOptimizer", "AdamOptimizer", "FtrlOptimizer",
	 "ProximalGradientDescentOptimizer", "RMSPropOptimizer"], default="AdamOptimizer")
parser.add_argument("-i", "--input", dest="IN", default=os.environ["RHORHO_DATA"])
parser.add_argument("-d", "--dropout", dest="DROPOUT", type=float, default=0.2)
parser.add_argument("-e", "--epochs", dest="EPOCHS", type=int, default=3)
parser.add_argument("-f", "--features", dest="FEAT", help="Features",
	choices=["Model-Oracle", "Model-OnlyHad", "Model-Benchmark", "Model-1", "Model-2", "Model-3"], default="Model-Oracle")
parser.add_argument("--treedepth", dest="TREEDEPTH", type=int, default=5)
parser.add_argument("--miniset", dest="MINISET", type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=False)
parser.add_argument("--svm_c", dest="SVM_C", type=float)
parser.add_argument("--svm_gamma", dest="SVM_GAMMA", type=float)
parser.add_argument("--forest_max_feat", dest="FOREST_MAX_FEAT", choices=["log2", "sqrt"], default="sqrt")
parser.add_argument("--forest_max_depth", dest="FOREST_MAX_DEPTH", default=10, type=int)
parser.add_argument("--forest_estimators", dest="FOREST_ESTIMATORS", default=10, type=int)
parser.add_argument("--z_noise_fraction", dest="Z_NOISE_FRACTION", type=float, default=0.5)
parser.add_argument("--unweighted", dest="UNWEIGHTED", type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=False)

args = parser.parse_args()

types[args.TYPE](args)
