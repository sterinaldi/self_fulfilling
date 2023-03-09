import numpy as np
import optparse as op
from pathlib import Path
from self_fulfilling.model import SF
from figaro.plot import plot_1d_dist

def main():
    
    parser = op.OptionParser()
    # Input/output
    parser.add_option("-i", "--input", type = "string", dest = "matrix_file", help = "Matrix file", default = None)
    parser.add_option("-o", "--output", type = "string", dest = "output", help = "Output folder. Default: same directory as matrix", default = None)
    parser.add_option("--draws", type = "int", dest = "n_draws", help = "Number of draws", default = 1000)
    
    (options, args) = parser.parse_args()
    # File
    options.matrix_file = Path(options.matrix_file).resolve()
    if options.output is not None:
        options.output = Path(options.output).resolve()
        if not options.output.exists():
            options.output.mkdir(parents=True)
    else:
        options.output = options.samples_file.parent
    
    name = options.matrix_file.parts[-1].split('.')[0]

    matrix = np.loadtxt(options.matrix_file)
    model = SF(matrix)
    draws = model.rvs(options.n_draws)
    
    plot_1d_dist(np.arange(1, model.P_max+1), draws, name = name, label = p)

if __name__ == '__main__':
    main()
