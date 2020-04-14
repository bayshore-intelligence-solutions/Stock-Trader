import yaml
from pathlib import Path


class Config:
    ROOT = None

    def __init__(self, config):
        with open(config, 'r') as stream:
            self._config = yaml.load(stream,
                                     Loader=yaml.CLoader)
            Config.ROOT = Path(self._config['root_path']).joinpath('src').joinpath('engine')
            if self._config['type'] not in ['many2one',
                                            'many2many']:
                raise NotImplementedError('Unknown LSTM architecture!!')
            else:
                self._arch = self._config[self._config['type']]

    @property
    def name(self):
        return self._config['type']

    @property
    def arch(self):
        return self._arch

    @property
    def data(self):
        return self._config['data']

    @property
    def lstm(self):
        return self._arch['lstm']

    @property
    def layers(self):
        return self._arch['layers']

    @property
    def ops(self):
        return self._arch['fit']

    @property
    def cell_dim(self):
        return self.lstm['time_step'], len(self.data['cols'])

    @property
    def tensorboard(self):
        return self._config['tensorboard']

    @property
    def root(self):
        return Path(self._config['root_path']).joinpath('src').joinpath('engine')

# if __name__ == '__main__':
#     conf = Config('config.yaml')
#     print(conf.arch)
#     print(conf.data)
#     print(conf.root)
