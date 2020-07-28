#!/usr/bin/env python3
import argparse
import yaml
import icebear


class Processing:
    def __init__(self, config):
        filenames = icebear.generate_level1(config)
        config.add_attr('level1_data_files', filenames)


class Imaging:
    def __init__(self, config):
        self.check_coeffs(config)
        pass

    def check_coeffs(self, config):
        if config.check_attr('swht_coeffs'):
            if config.compare_attr('swht_coeffs', None):
                filename = icebear.generate_coeffs(config)
                config.add_attr('swht_coeff', filename)
            pass
        else:
            print(f'ERROR: Attribute {key} does not exists')
        return None


class Plotting:
    def __init__(self, config):
        
        

class Config:
    def __init__(self, configuration, settings):
        self.cfg_file = configuration
        self.set_file = settings
        with open(self.cfg_file, 'r') as stream:
            cfg = yaml.full_load(stream)
            for key, value in cfg.items():
                setattr(self, key, value)
        if self.set_file:
            with open(self.set_file, 'r') as stream:
                cfg = yaml.full_load(stream)
                for key, value in cfg.items():
                    setattr(self, key, value)

    def print_attrs(self):
        print("Experiment attributes loaded: ")
        for item in vars(self).items():
            print(f"\t-{item}")
        return None

    def update_attr(self, key, value):
        if not self.check_attr(key):
            print(f'ERROR: Attribute {key} does not exists')
            exit()
        else:
            setattr(self, key, value)
        return None

    def check_attr(self, key):
        if hasattr(self, key):
            return True
        else:
            return False

    def compare_attr(self, key, value):
        if not self.check_attr(key):
            print(f'ERROR: Attribute {key} does not exists')
            exit()
        else:
            if getattr(self, key) == value:
                return True
            else:
                return False

    def add_attr(self, key, value):
        if self.check_attr(key):
            print(f'ERROR: Attribute {key} already exists')
            exit()
        else:
            setattr(self, key, value)
        return None

    def remove_attr(self, key):
        if not self.check_attr(key):
            print(f'ERROR: Attribute {key} does not exists')
            exit()
        else:
            delattr(self, key)
        return None


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
    config = Config(args.configuration, args.settings)
    config.print_attrs()

    if args.processing:
        if args.start:
            config.update_attr('processing_start', args.start)
        if args.stop:
            config.update_attr('processing_stop', args.stop)
        if args.step:
            config.update_attr('processing_step', args.step)
        Processing(config)

    if args.imaging:
        if args.start:
            config.update_attr('imaging_start', args.start)
        if args.stop:
            config.update_attr('imaging_stop', args.stop)
        if args.step:
            config.update_attr('imaging_step', args.step)
        Imaging(config)

    if args.plotting:
        if args.start:
            config.update_attr('plotting_start', args.start)
        if args.stop:
            config.update_attr('plotting_stop', args.stop)
        if args.step:
            config.update_attr('plotting_step', args.step)
        Plotting(config)

    return None

if __name__ == '__main__':
    main()
