import os
from dotenv import load_dotenv

load_dotenv()
LEVIR_LOCATION = os.getenv("LEVIR_DATASET_LOCATION")
WHU_LOCATION = os.getenv("WHU_DATASET_LOCATION")
CDD_LOCATION = os.getenv("CDD_DATASET_LOCATION")

class DataConfig:
    data_name = ""
    root_dir = ""
    label_transform = ""
    def get_data_config(self, data_name):
        self.data_name = data_name
        if data_name == 'LEVIR':
            self.label_transform = "norm"
            self.root_dir = LEVIR_LOCATION
       
        elif data_name == 'WHU':
            self.label_transform = "norm"
            self.root_dir = WHU_LOCATION

        elif data_name == 'CDD':
            self.label_transform = "norm"
            self.root_dir = CDD_LOCATION

        else:
            raise TypeError('%s has not defined' % data_name)
        return self


if __name__ == '__main__':
    data = DataConfig().get_data_config(data_name='LEVIR')
    print(data.data_name)
    print(data.root_dir)
    print(data.label_transform)

