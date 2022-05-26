from dataclasses import dataclass


@dataclass
class Validator:
    def _validate_range(self, key, value):
        min = self.__dataclass_fields__[key].metadata['min']
        max = self.__dataclass_fields__[key].metadata['max']
        if value and not min <= value <= max:
            raise ValueError(f'Attribute {key}:{value} is not between {min} and {max}')

    def _validate_choices(self, key, value):
        options = self.__dataclass_fields__[key].metadata['options']
        if options and value not in options:
            raise ValueError(f'Attribute {key}:{value} is not an expected option:{options}')

    def _validate_type(self, key, value):
        type = self.__dataclass_fields__[key].default_factory
        if not isinstance(value, type):
            raise ValueError(f'Attribute {key}:{type(value)} is not the expected type:{type}')

    def _validate_shape(self, key, value):
        shape = self.__dataclass_fields__[key].metadata['shape']
        if value.shape is not shape:
            raise ValueError(f'Attribute {key}{value.shape} is not the expected shape:{shape}')