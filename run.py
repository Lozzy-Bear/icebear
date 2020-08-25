#!/usr/bin/env python3
import argparse
import icebear.utils
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
        time = icebear.utils.Time(config.plotting_start, config.plotting_stop, config.plotting_step)
        icebear.quick_look(config, time)
        pass


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
    #if (args.processing or args.imaging or args.plotting) and not args.settings:
    #    print('ERROR: No settings file -s set to configure -prc, -img, or -plt options')
    #    exit()
    args.func(args)


def run(args):
    config = icebear.utils.Config(args.configuration, args.settings)
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
