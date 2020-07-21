#!/usr/bin/env python3
import argparse
import yaml
import icebear


class Config:
    def __init__(self, args):
        self.cfg_file = args.configuration
        self.set_file = args.settings
        with open(self.cfg_file, 'r') as stream:
            cfg = yaml.full_load(stream)
            for key, value in cfg.items():
                setattr(self, key, value)
        if self.set_file:
            with open(self.set_file, 'r') as stream:
                cfg = yaml.full_load(stream)
                for key, value in cfg.items():
                    setattr(self, key, value)


def main():
    parser = argparse.ArgumentParser(description="Setup icebear configuration.")
    parser.add_argument("-c", "--configuration", help="configuration file, leave blank for default",
                        type=str, default='dat/default.yml')
    parser.add_argument("-s", "--settings", help="optional settings file, overrides -c attributes",
                        type=str, required=False)
    parser.add_argument("-prc", "--processing", help='run the processing class',
                        action='store_true')
    parser.add_argument("-img", "--imaging", help='run the imaging class',
                        action='store_true')
    parser.add_argument("-plt", "--plotting", help='run the plotting class',
                        action='store_true')
    parser.add_argument("--start", type=int, required=False,
                        help="data start point, [year, month, day, hour, minute, second], overrides -c and -s")
    parser.add_argument("--stop", type=int, required=False,
                        help="data stop point, [year, month, day, hour, minute, second], overrides -c and -s")
    parser.add_argument("--step", type=int, required=False,
                        help="data step size, [year, month, day, hour, minute, second], overrides -c and -s")
    parser.set_defaults(func=run)
    args = parser.parse_args()
    if (args.processing or args.imaging or args.plotting) and not args.settings:
        print('ERROR: No settings file -s set to configure -prc, -img, or -plt options')
        exit()
    args.func(args)


def run(args):
    config = Config(args)
    print("Experiment attributes loaded: ")
    for item in vars(config).items():
        print(f"\t-{item}")

    if args.processing:
        Proccesing(config)
    if args.imaging:
        icebear.generate_coeffs(config)
    if args.plotting:
        Plotting(config)


if __name__ == '__main__':
    main()
