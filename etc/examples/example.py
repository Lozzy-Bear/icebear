import icebear
import icebear.utils as utils

"""
Example : level 1 processing with default configuration

    By design there is two strong couplings through out the package. That is Config and Time classes.
    This is used to ensure the metadata for any computation done is stream lined as it all comes from two places.
"""

# (configuration file, settings file)
config = utils.Config('dat/default.yml', '')
# [year, month, day, hour, minute, second, microsecond]
config.update_attr('processing_start', [2019, 11, 20, 3, 0, 0, 0])
# [year, month, day, hour, minute, second, microsecond]
config.update_attr('processing_stop', [2019, 11, 20, 23, 0, 0, 0])
# [day, hour, minute, second, microsecond]
config.update_attr('processing_step', [0, 0, 0, 1, 0])
config.update_attr('snr_cutoff', 3.0)                        # Set snr cutoff to 3.0 dB for processing
config.add_attr(key='key', value='value')                    # We can add/remove attributes also

icebear.generate_level1(config)                              # Run the processing function
