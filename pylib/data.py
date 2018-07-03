#load the data feeding to model
class HSFeeder:
    def __init__(self,params):
        pass
    
    def generate_batch(self):
        while True:
            batch_data = self._get_batch_data()
            yield batch_data

    def _get_batch_data(self):
        batch_data = dict()
        return batch_data